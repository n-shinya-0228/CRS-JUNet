"""Single-pass C.2 recipe, identical raw pipeline for the two E.1 arms."""

from collections import Counter
from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from generate_point_features import load_encoder
from lib.dataset.Parser_PointNet import bev_collate_fn
from lib.dataset.SemanticKitti_PointNet import SemanticKitti as CachedValidation
from lib.dataset.SemanticKitti_PointNet_RawCopyPaste_E1 import SemanticKitti, raw_collate
from lib.losses import Lovasz_softmax
from lib.models.SJNet3_PointNet_Gated_Local3 import SJNet3_PointNet_Gated_Local3
from lib.models.SJNet3_PointNet_Gated_Local3_ClassAlpha import CLASS_NAMES
from lib.utils.ioueval import iouEval
from lib.utils.pointnet_c2_recipe_e1 import C2Recipe, EarlyStopping, FocalLoss, SobelFilter, build_scheduler
from lib.utils.pointnet_copy_paste_e0 import sha256
from lib.utils.raw_point_copy_paste_e0b import apply_dropout, regenerate_hybrid
from lib.utils.raw_point_copy_paste_e0c import ENCODER_SHA256, stable_seed


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temporary, path)


def atomic_checkpoint(path, state):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def deterministic_encoder():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    precision = torch.get_float32_matmul_precision()
    # The cached frozen encoder used full FP32; C.2 segmentation uses "high" (TF32).
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        torch.set_float32_matmul_precision(precision)


def augmentation(result, seed):
    generator = torch.Generator().manual_seed(seed)
    x, labels, paste = result["hybrid24"], result["label"], result["paste_mask"]
    shift = int(torch.randint(0, 512, (1,), generator=generator))
    x, labels, paste = torch.roll(x, shift, -1), torch.roll(labels, shift, -1), torch.roll(paste, shift, -1)
    if torch.rand(1, generator=generator).item() > .5:
        x, labels, paste = x.flip(-1), labels.flip(-1), paste.flip(-1)
    if torch.rand(1, generator=generator).item() > .5:
        keep = torch.rand((512, 512), generator=generator) > .10
        x, labels, paste = apply_dropout(x, labels, paste, keep, policy="legacy_denoising")
    return x, labels, paste


def validation_dataset(root, data, arch):
    if data["split"]["valid"] != [8]:
        raise ValueError("Require sequence 08 validation")
    result = CachedValidation(root, [8], data["labels"], data["color_map"], data["learning_map"],
                              data["learning_map_inv"], arch["dataset"]["sensor"], is_train=False,
                              copy_paste_prob=0., pointnet_folder="pointnet_16ch", pointnet_feature_dim=16)
    if len(result) != 4071:
        raise ValueError("Require all 4071 validation scans")
    return result


def class_weights(data, epsilon):
    content = torch.zeros(20)
    for raw, frequency in data["content"].items():
        content[data["learning_map"][raw]] += frequency
    weights = (1/(content/(content.sum()+1e-6)+epsilon)).clamp(max=20.)
    for cid, ignore in data["learning_ignore"].items():
        if ignore:
            weights[cid] = 0
    return weights


def append_csv(path, row):
    path = Path(path)
    exists = path.exists()
    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k,v in row.items()})


class Trainer(C2Recipe):
    def __init__(self, arch, data, root, output, logger, enabled=False, resume=None):
        self.arch = arch
        self.data = data
        self.output = Path(output)
        self.logger = logger
        self.enabled = bool(enabled)
        self.device = torch.device("cuda:0")
        if not torch.cuda.is_available():
            raise RuntimeError("Real E.1 execution requires CUDA")
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision("high")
        torch.cuda.set_per_process_memory_fraction(.90)
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        # Initialization precedes loading the fixed encoder in BOTH experiments.
        self.model = SJNet3_PointNet_Gated_Local3(in_channels=24, nclasses=20, drop=.3).to(self.device)
        self.initial_hash = state_digest(self.model.state_dict())
        self.use_ema, self.ema_decay = True, .99
        self._build_ema_model()
        self.ema_model.eval()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=arch["train"]["lr"],
                                           weight_decay=arch["train"]["w_decay"])
        self.scheduler = build_scheduler(self.optimizer, arch["train"])
        self.scaler = torch.amp.GradScaler("cuda")
        self.loss_w = class_weights(data, arch["train"]["epsilon_w"])
        self.criterion_main = FocalLoss(self.loss_w, gamma=1., ignore_index=0).to(self.device)
        self.lovasz = Lovasz_softmax(ignore=0).to(self.device)
        self.sobel = SobelFilter().to(self.device)
        self.boundary_criterion = nn.BCEWithLogitsLoss(reduction="none").to(self.device)
        self.set_epoch_weights(0)
        encoder_path = arch["dataset"]["pointnet"]["encoder"]
        if sha256(encoder_path) != ENCODER_SHA256:
            raise ValueError("Frozen PointNet encoder hash mismatch")
        self.encoder, _ = load_encoder(encoder_path, 16, self.device)
        self.encoder.eval().requires_grad_(False)
        self.encoder_state_hash = state_digest(self.encoder.state_dict())
        self.dataset = SemanticKitti(root, data, arch["copy_paste"], enabled)
        self.validation = validation_dataset(root, data, arch)
        self.workers = arch["train"]["workers"]
        self.early_stopping = EarlyStopping(**arch["train"]["early_stopping"])
        self.best_val_iou, self.best_val_epoch, self.next_epoch, self.next_batch = -1., -1, 0, 0
        self.history = []
        self.partial = None
        self.interrupted = False
        self.elapsed_seconds = 0.
        source_paths = ["lib/trainer/trainer_PointNet_RawCopyPaste_E1.py", "lib/dataset/SemanticKitti_PointNet_RawCopyPaste_E1.py",
                        "lib/utils/raw_point_copy_paste_e0c.py", "lib/utils/raw_point_copy_paste_e0b.py",
                        "lib/utils/pointnet_c2_recipe_e1.py", "lib/models/SJNet3_PointNet_Gated_Local3.py"]
        train_manifest=[]
        for path in self.dataset.files:
            label=path.parent.parent/"labels"/(path.stem+".label")
            train_manifest.append([dict(path=str(p.resolve()),size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in (path,label)])
        from evaluate_pointnet_fixed_alpha_sensitivity import dataset_manifest
        self.validation_manifest_sha256=dataset_manifest(self.validation,root)[2]
        self.identity = dict(arch=arch, data=data, enabled=self.enabled,
                             raw_train_manifest=train_manifest,validation_manifest_sha256=self.validation_manifest_sha256,
                             db_manifest=sha256(Path(arch["copy_paste"]["donor_database"])/"manifest.json"),
                             encoder=ENCODER_SHA256, initial_state_sha256=self.initial_hash,
                             code={p:sha256(p) for p in source_paths})
        if resume:
            self.load(resume)

    def set_epoch_weights(self, epoch):
        ratio = 1. if epoch < 50 else max(0., (60-epoch)/10.)
        self.w_aux2, self.w_aux4, self.w_aux8 = .30*ratio, .15*ratio, .10
        self.w_lovasz, self.w_boundary, self.w_paste = .50, .20, 0.

    def loader(self, epoch, start_batch=0):
        self.dataset.epoch = epoch
        order = torch.randperm(len(self.dataset), generator=torch.Generator().manual_seed(stable_seed(epoch, 0, "scan_order")))
        order = order[:len(order)//3*3].tolist()[start_batch*3:]
        generator = torch.Generator().manual_seed(stable_seed(epoch, 0, "workers"))
        return DataLoader(self.dataset, batch_size=3, sampler=order, drop_last=True, num_workers=self.workers,
                          collate_fn=raw_collate, generator=generator,
                          **(dict(prefetch_factor=2) if self.workers else {}))

    def val_loader(self):
        return DataLoader(self.validation, batch_size=3, shuffle=False, num_workers=self.workers,
                          collate_fn=bev_collate_fn, pin_memory=True)

    def batch_input(self, batch):
        inputs, targets, pastes = [], [], []
        for sample in batch:
            with deterministic_encoder():
                result = regenerate_hybrid(sample["points"], sample["packed"], self.encoder,
                                           self.data["learning_map"], sample["paste_points"])
                if sample["before_points"] is not None:
                    from lib.utils.laserscan_Pointnet import build_polar_point_features
                    _, before_cells, _ = build_polar_point_features(torch.from_numpy(sample["before_points"]).to(self.device))
                    sample["stats"]["new_observed_cells"] = int(np.setdiff1d(result["cell_ids"].numpy(), before_cells.cpu().numpy()).size)
            sample["stats"]["regeneration_seconds"] = result["timing"]["total_seconds"]
            x, y, paste = augmentation(result, sample["augmentation_seed"])
            inputs.append(x)
            targets.append(y)
            pastes.append(paste)
        x = torch.stack(inputs).to(self.device)
        assert x.shape == (len(batch), 24, 512, 512)
        return x, torch.stack(targets).to(self.device), torch.stack(pastes).to(self.device)

    def boundary(self, labels, mask):
        return self._compute_boundary_gt(labels)*mask

    def update(self, batch):
        self.model.train()
        x, labels, paste = self.batch_input(batch)
        self.optimizer.zero_grad(set_to_none=True)
        boundary = self.boundary(labels, x[:,23:24])
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(x)
            loss = self._mix_losses(outputs, labels, boundary, x[:,23:24], paste)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
        if not torch.isfinite(norm):
            raise RuntimeError("Nonfinite segmentation gradients")
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self._update_ema()
        if self.encoder.training or any(p.grad is not None or p.requires_grad for p in self.encoder.parameters()):
            raise RuntimeError("Frozen encoder has train mode/gradients")
        from evaluate_pointnet_fixed_alpha_sensitivity import original_predictions
        predictions = original_predictions(outputs["logits"].detach(), x[:,:23])
        return float(loss.detach()), predictions, labels, dict(gradient_norm=float(norm),
                    outputs={k:list(v.shape) for k,v in outputs.items()}, input_shape=list(x.shape))

    @torch.no_grad()
    def validate(self):
        self.ema_model.eval()
        evaluator = iouEval(20, self.device, [0])
        loss_sum, scans = 0., 0
        from evaluate_pointnet_fixed_alpha_sensitivity import original_predictions
        for batch in self.val_loader():
            x = torch.cat((batch[0],batch[1].float()), 1).to(self.device)
            labels = batch[2].long().to(self.device)
            outputs = self.ema_model(x)  # C.2 validation is FP32, without autocast.
            loss = self._mix_losses(outputs, labels, self.boundary(labels,x[:,23:24]), x[:,23:24])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite validation loss")
            loss_sum += float(loss)*len(x)
            scans += len(x)
            evaluator.addBatch(original_predictions(outputs["logits"],x[:,:23]), labels)
        if scans != 4071:
            raise RuntimeError("Incomplete validation")
        miou, per_class = evaluator.getIoU()
        return dict(miou=float(miou), accuracy=float(evaluator.getacc()), loss=loss_sum/scans, scans=scans,
                    class_iou={CLASS_NAMES[cid]:float(per_class[cid]) for cid in range(1,20)})

    def save(self, epoch, next_batch=0, partial=None, best=False, elapsed_seconds=None):
        state = dict(experiment="Step E.1", identity=self.identity, arch_cfg=self.arch, data_cfg=self.data,
                     epoch=epoch, next_epoch=epoch if next_batch else epoch+1, next_batch=next_batch,
                     model_train=self.model.state_dict(), model_ema=self.ema_model.state_dict(),
                     model=self.ema_model.state_dict() if best else self.model.state_dict(), optim=self.optimizer.state_dict(),
                     scheduler=self.scheduler.state_dict(), scaler=self.scaler.state_dict(),
                     early_stopping=self.early_stopping.state_dict(), best_val_iou=self.best_val_iou,
                     best_val_epoch=self.best_val_epoch, history=self.history,
                     elapsed_seconds=self.elapsed_seconds if elapsed_seconds is None else elapsed_seconds,
                     partial=partial, rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                                            torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()))
        path = self.output/("best_val.path" if best else "latest.path")
        atomic_checkpoint(path, state)
        self.logger.info(f"Saved checkpoint: {path} (epoch {epoch}, next batch {next_batch})")

    def load(self, path):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["identity"] != self.identity or saved.get("experiment") != "Step E.1":
            raise ValueError("E.1 checkpoint identity mismatch")
        self.model.load_state_dict(saved["model_train"], strict=True)
        self.ema_model.load_state_dict(saved["model_ema"], strict=True)
        self.optimizer.load_state_dict(saved["optim"])
        self.scheduler.load_state_dict(saved["scheduler"])
        self.scaler.load_state_dict(saved["scaler"])
        self.early_stopping.load_state_dict(saved["early_stopping"])
        for key in ("next_epoch", "next_batch", "best_val_iou", "best_val_epoch", "history", "elapsed_seconds", "partial"):
            setattr(self, key, saved[key])
        random.setstate(saved["rng"]["python"])
        np.random.set_state(saved["rng"]["numpy"])
        torch.set_rng_state(saved["rng"]["torch"])
        torch.cuda.set_rng_state_all(saved["rng"]["cuda"])
        self.set_epoch_weights(self.next_epoch)
        self.logger.info(f"Resumed {path}: epoch {self.next_epoch}, batch {self.next_batch}")

    def train(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: setattr(self, "interrupted", True))
        if not (self.output/"epoch0_random_init.path").exists():
            atomic_checkpoint(self.output/"epoch0_random_init.path", dict(model=self.model.state_dict(), seed=0,
                                state_sha256=self.initial_hash, scratch=True))
        if self.early_stopping.should_stop(self.next_epoch-1):
            return
        for epoch in range(self.next_epoch, self.arch["train"]["max_epochs"]):
            begin = time.perf_counter()
            timing_start = time.process_time()
            self.set_epoch_weights(epoch)
            rate = self.optimizer.param_groups[0]["lr"]
            self.logger.info(f"Epoch {epoch:02d} | Learning rate: {rate:.6e}")
            evaluator = iouEval(20,self.device,[0])
            meters = dict(loss_sum=0., accuracy_sum=0., iou_sum=0., scans=0, cp=[],peak_allocated_GiB=0.)
            if self.partial:
                meters = self.partial["meters"]
                evaluator.conf_matrix.copy_(torch.tensor(self.partial["confusion"],device=self.device))
            prior_epoch_seconds = self.partial.get("epoch_seconds",0.) if self.partial else 0.
            start_batch = self.next_batch
            for index, batch in enumerate(self.loader(epoch,start_batch), start_batch):
                loss, predictions, labels, diagnostics = self.update(batch)
                meters["peak_allocated_GiB"] = max(meters["peak_allocated_GiB"],torch.cuda.max_memory_allocated()/2**30)
                evaluator.addBatch(predictions, labels)
                accuracy = float(evaluator.getacc())
                miou = float(evaluator.getIoU()[0])
                meters["scans"] += 3
                for key, value in (("loss_sum",loss),("accuracy_sum",accuracy),("iou_sum",miou)):
                    meters[key] += value*3
                meters["cp"].extend(s["stats"] for s in batch)
                if index % 3000 == 0:
                    n = meters["scans"]
                    self.logger.info(f"Lr: {rate:.3e} | Epoch: [{epoch}][{index}/{len(self.dataset)//3}] | "
                                     f"Loss {loss:.4f} ({meters['loss_sum']/n:.4f}) | "
                                     f"acc {accuracy:.3f} ({meters['accuracy_sum']/n:.3f}) | "
                                     f"IoU {miou:.3f} ({meters['iou_sum']/n:.3f})")
                    try:
                        snapshot=subprocess.run(["nvidia-smi","--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total",
                                                 "--format=csv"],capture_output=True,text=True,timeout=5,check=True)
                        with (self.output/"used_files.txt").open("a") as file:
                            file.write(f"Epoch {epoch} batch {index}, point-in-time GPU sample (not epoch average):\n"+snapshot.stdout)
                            file.write(f"Main-process CPU usage since epoch/resume segment: "
                                       f"{100*(time.process_time()-timing_start)/(time.perf_counter()-begin):.1f}% "
                                       "(one-core scale; workers excluded)\n")
                    except (OSError,subprocess.SubprocessError):
                        pass
                if self.interrupted:
                    self.elapsed_seconds += time.perf_counter()-begin
                    self.save(epoch,index+1,dict(meters=meters,confusion=evaluator.conf_matrix.cpu().tolist(),
                              epoch_seconds=prior_epoch_seconds+time.perf_counter()-begin))
                    atomic_json(self.output/"results.json",dict(status="interrupted",epoch=epoch,batch=index+1,
                                best_epoch=self.best_val_epoch,best_miou=self.best_val_iou))
                    self.logger.info("Stopped by request; optimizer-boundary checkpoint saved")
                    return
                if index % 3000 == 0:
                    self.save(epoch,index+1,dict(meters=meters,confusion=evaluator.conf_matrix.cpu().tolist(),
                              epoch_seconds=prior_epoch_seconds+time.perf_counter()-begin),
                              elapsed_seconds=self.elapsed_seconds+time.perf_counter()-begin)
            self.next_batch, self.partial = 0, None
            self.scheduler.step()
            result = self.validate()
            stop = self.early_stopping.update(result["miou"],epoch)
            improved = result["miou"] > self.best_val_iou
            if improved:
                self.best_val_iou, self.best_val_epoch = result["miou"], epoch
            stats = summarize_cp(meters["cp"])
            self.logger.info(f"Copy-Paste | Attempts: {stats['attempted_scans']} | Successful scans: {stats['successful_scans']} "
                             f"| Success rate: {stats['success_rate']:.2%} | Classes: {stats['class_successes']}")
            self.logger.info(f"Validation epoch {epoch}: loss {result['loss']:.4f} | acc {result['accuracy']:.6f} | mIoU {result['miou']:.9f}")
            for name,value in result["class_iou"].items():
                self.logger.info(f"IoU class [{name}] = {value:.6f}")
            self.logger.info(f"Best Validation mIoU: {self.best_val_iou:.9f} | Best Epoch: {self.best_val_epoch} | "
                             f"Early Stopping: {self.early_stopping.bad_epochs}/12 (min epoch 30) | Lr: {rate:.6e}")
            self.elapsed_seconds += time.perf_counter()-begin
            record = dict(epoch=epoch,lr=rate,training_loss=meters["loss_sum"]/meters["scans"],
                          training_acc=meters["accuracy_sum"]/meters["scans"],training_iou=meters["iou_sum"]/meters["scans"],
                          **result,best_miou=self.best_val_iou,best_epoch=self.best_val_epoch,
                          early_stopping=self.early_stopping.state_dict(),epoch_seconds=prior_epoch_seconds+time.perf_counter()-begin,
                          peak_allocated_GiB=meters["peak_allocated_GiB"],
                          main_process_cpu_percent=100*(time.process_time()-timing_start)/(time.perf_counter()-begin))
            self.history.append(record)
            atomic_json(self.output/"training_history.json",self.history)
            append_csv(self.output/"validation_history.csv",{k:v for k,v in record.items() if k != "class_iou"})
            append_csv(self.output/"class_iou_history.csv",dict(epoch=epoch,**result["class_iou"]))
            append_csv(self.output/"cp_stats.csv",dict(epoch=epoch,**stats))
            append_csv(self.output/"copy_paste_statistics.csv",dict(epoch=epoch,**stats))
            with (self.output/"used_files.txt").open("a") as file:
                file.write(f"Epoch {epoch}: seconds={record['epoch_seconds']:.3f}, GPU allocated peak={record['peak_allocated_GiB']:.3f} GiB, "
                           f"main-process CPU={record['main_process_cpu_percent']:.1f}% (one-core scale; workers excluded), "
                           f"raw regeneration seconds summed={stats['regeneration_seconds']:.3f}, "
                           f"worker seconds summed={stats['worker_seconds']:.3f} (overlapped, not additive wall time)\n")
            if improved:
                self.save(epoch,best=True)
            self.save(epoch)
            atomic_json(self.output/"results.json",dict(status="complete" if stop or epoch==69 else "running",
                         completed_epoch=epoch,best_epoch=self.best_val_epoch,best_miou=self.best_val_iou,
                         early_stopping=self.early_stopping.state_dict(),training_seconds=self.elapsed_seconds))
            if stop:
                self.logger.info(f"Early stopping triggered: {self.early_stopping.bad_epochs} epochs without >=0.001 improvement; min epoch 30 reached")
                break
        self.logger.info("Finished Training")


def summarize_cp(rows):
    result = {key:sum(r[key] for r in rows) for key in ("scans","attempted_scans","successful_scans","fallback",
              "candidate_trials","added_points","removed_points","new_observed_cells","worker_seconds","regeneration_seconds")}
    for key in ("class_attempts","class_successes","rejection_reasons"):
        counter = Counter()
        for row in rows:
            counter.update(row[key])
        result[key] = dict(counter)
    for key in ("labels_before","labels_after"):
        result[key] = np.asarray([r[key] for r in rows],dtype=np.int64).sum(0).tolist()
    result["rare_class_point_delta"] = {CLASS_NAMES[cid]:result["labels_after"][cid]-result["labels_before"][cid] for cid in range(2,9)}
    result["success_rate"] = result["successful_scans"]/max(1,result["attempted_scans"])
    return result
