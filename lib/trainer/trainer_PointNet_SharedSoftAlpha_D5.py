"""D.5 scratch training, C.2 losses/schedule and effective-batch log format."""

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from lib.dataset.Parser_PointNet import Parser, bev_collate_fn
from lib.losses import Lovasz_softmax
from lib.models.SJNet3_PointNet_Gated_Local3 import SJNet3_PointNet_Gated_Local3
from lib.models.SJNet3_PointNet_Gated_Local3_ClassAlpha import ALPHA_TABLE, CLASS_NAMES, build_alpha_maps
from lib.models.SJNet3_PointNet_Gated_Local3_SharedSoftAlpha import SJNet3_PointNet_Gated_Local3_SharedSoftAlpha
from lib.utils.avgmeter import AverageMeter
from lib.utils.pointnet_class_alpha_statistics import AlphaStatistics, HIST_WIDTH
from lib.utils.pointnet_d4_training import EarlyStopping, build_scheduler

ROOT = Path(__file__).resolve().parents[2]
C2_CONFIG = ROOT/"config/arch/sjnet_pointnet_gated_local3_stablelr.yaml"
_spec = importlib.util.spec_from_file_location("d5_readonly_d4_helpers", Path(__file__).with_name("trainer_PointNet_TeacherStudent_D4.py"))
d4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d4)
d1, d1b = d4.d1, d4.d1b


def validate_config(arch):
    c2 = yaml.safe_load(C2_CONFIG.read_text())
    if arch["model"]["name"] != "SJNet3_PointNet_Gated_Local3_SharedSoftAlpha":
        raise ValueError("D.5 requires its own scratch shared-weight model")
    for section in ("dataset", "post"):
        if arch[section] != c2[section]:
            raise ValueError(f"D.5 must preserve the C.2 {section} configuration")
    for key, value in c2["train"].items():
        if arch["train"].get(key) != value:
            raise ValueError(f"D.5 must preserve C.2 train.{key}={value}")
    for key in ("in_channels", "dropout", "bn_d", "use_mps"):
        if arch["model"].get(key) != c2["model"][key]:
            raise ValueError(f"D.5 must preserve C.2 model.{key}")
    train = arch["train"]
    if (train.get("seed") != 0 or train.get("micro_batch_size") not in (1, 2, 3)
            or train.get("use_ema") is not True or train.get("ema_decay") != .99
            or train.get("amp") is not True or train.get("amp_dtype") != "bfloat16"):
        raise ValueError("Require seed 0, C.2 BF16 + EMA .99, and microbatch 1/2/3")
    if arch.get("experiment", {}).get("pretrained_initialization") is not False:
        raise ValueError("Pretrained initialization is forbidden in D.5")


def configure_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("D.5 real-data execution requires CUDA")
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(.85, device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.fastest = True
    return device


def should_report(batch_index, report=3000):
    return batch_index % report == 0


def log_training_progress(logger, epoch, batch_index, total, lr, loss, accuracy, iou):
    logger.info(
        "Lr: {lr:.3e} | Epoch: [{ep}][{it}/{tot}] | Loss {lcur:.4f} ({lavg:.4f}) | "
        "acc {acur:.3f} ({aavg:.3f}) | IoU {icur:.3f} ({iavg:.3f})".format(
            lr=lr, ep=epoch, it=batch_index, tot=total, lcur=loss.val, lavg=loss.avg,
            acur=accuracy.val, aavg=accuracy.avg, icur=iou.val, iavg=iou.avg))


def log_validation_summary(logger, epoch, lr, result, best, best_epoch, stopping):
    first, second = result["stage1"], result["stage2"]
    logger.info("*"*80)
    logger.info(f"Validation | Epoch: [{epoch}] | Stage 1 mIoU: {first['miou']:.9f} | Stage 1 Accuracy: {first['accuracy']:.9f}")
    logger.info(f"Validation | Epoch: [{epoch}] | Stage 2 mIoU: {second['miou']:.9f} | Stage 2 Accuracy: {second['accuracy']:.9f} | Stage 2 - Stage 1: {second['miou']-first['miou']:+.9f} | Stage 2 Loss: {second['loss']:.6f}")
    for stage in ("stage1", "stage2"):
        for cid in range(1, 20):
            name = CLASS_NAMES[cid]
            logger.info(f"Stage {stage[-1]} | IoU class {cid:2d} [{name}] = {result[stage]['per_class_iou'][name]:.6f}")
    stats = result["alpha_statistics"]["observed"]
    logger.info(f"Soft alpha (observed) | mean: {stats['mean']:.6f} | std: {stats['std']:.6f} | min: {stats['min']:.6f} | max: {stats['max']:.6f}")
    logger.info(f"Best Validation mIoU: {best:.9f} | Best Epoch: {best_epoch} | Learning Rate: {lr:.6e}")
    reference = stopping.best_score
    logger.info(f"Early stopping: current mIoU={second['miou']:.6f} | best mIoU={best:.6f} | best epoch={best_epoch} | bad epochs={stopping.bad_epochs}/{stopping.patience} | significant-improvement reference={reference:.6f}")


def validate_resume(saved, arch, data, identity):
    if (saved.get("experiment") != "Step D.5" or saved.get("arch_cfg") != arch
            or saved.get("data_cfg") != data or saved.get("metadata_id") != identity
            or saved.get("alpha_table") != list(ALPHA_TABLE) or saved.get("copy_paste_prob") != 0
            or saved.get("pretrained_initialization") is not False
            or not saved.get("initialization_verified")):
        raise ValueError("Resume requires a matching native D.5 scratch checkpoint/config/source/table")


def nested_digest(value):
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(str((item.dtype, item.shape)).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            for entry in item:
                visit(entry)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


class Trainer(d4.C2LossEMATrainer):
    def __init__(self, arch, data, dataset, output, logger, resume=None, disposable=False):
        validate_config(arch)
        self.ARCH, self.DATA, self.output, self.logger = arch, data, Path(output), logger
        self.output.mkdir(parents=True, exist_ok=True)
        self.device = configure_cuda()
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        self.parser = Parser(root=dataset, data_cfg=data, arch_cfg=arch, gt=True, shuffle_train=False)
        source = ROOT/"logs/SJNet3_PointNet_stepD2_class_fixed_alpha"
        self.reference_run = json.loads((source/"results.json").read_text())
        self.reference = self.reference_run["results"]["D2_Soft"]
        if self.reference_run["status"] != "complete":
            raise ValueError("The D.2 reference must be complete")
        classes = {i:self.parser.get_xentropy_class_string(i) for i in range(20)}
        d4.validate_alpha_table(ROOT/"logs/SJNet3_PointNet_stepD1b_high_alpha_sensitivity", classes)
        entries, self.scan_ids, manifest_hash = d1.dataset_manifest(self.parser.valid_dataset, dataset)
        ref = self.reference_run["metadata"]
        if (len(self.scan_ids) != 4071 or manifest_hash != ref["validation_manifest_sha256"]
                or d1b.json_normalized(data) != ref["data_cfg"]
                or dict(entries=entries, scan_ids=self.scan_ids) != json.loads((source/"validation_manifest.json").read_text())
                or self.parser.train_dataset.copy_paste_prob != 0 or self.parser.valid_dataset.copy_paste_prob != 0
                or not self.parser.train_dataset.is_train or self.parser.valid_dataset.is_train):
            raise ValueError("D.5 must preserve original splits/PointNet inputs/mapping/CP=0")
        self.validation_entries = entries
        options = dict(in_channels=24, nclasses=20, drop=arch["model"]["dropout"])
        # This is a RANDOM C.2 initialization, never a trained C.2 checkpoint.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            temporary = SJNet3_PointNet_Gated_Local3(**options)
            c2_random_hash = d1.state_digest(temporary.state_dict())
            del temporary
        self.model = SJNet3_PointNet_Gated_Local3_SharedSoftAlpha(**options)
        self.initial_hash = d1.state_digest(self.model.state_dict())
        if (self.initial_hash != c2_random_hash or self.initial_hash == ref["model_state_sha256"]
                or sum(p.numel() for p in self.model.parameters()) != 28876586):
            raise ValueError("D.5 must match seed-0 C.2 random initialization, not its trained state")
        self.model.to(self.device)
        self.use_ema, self.ema_decay = True, .99
        self._build_ema_model()
        self.ema_model.eval()
        content = torch.zeros(20)
        for cid, frequency in data["content"].items():
            content[self.parser.to_xentropy(cid)] += frequency
        self.loss_w = torch.clamp(1/(content/(content.sum()+1e-6)+arch["train"]["epsilon_w"]), max=20.)
        for cid, ignored in data["learning_ignore"].items():
            if ignored:
                self.loss_w[cid] = 0
        self.criterion_main = d4._base.FocalLoss(weight=self.loss_w, gamma=1., ignore_index=0).to(self.device)
        self.lovasz = Lovasz_softmax(ignore=0).to(self.device)
        self.boundary_criterion = nn.BCEWithLogitsLoss(reduction="none").to(self.device)
        self.sobel = d4._base.SobelFilter().to(self.device)
        self.w_aux8, self.w_lovasz, self.w_boundary, self.w_paste = .10, .50, .20, 0.
        self.set_epoch_loss_weights(0)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=arch["train"]["lr"], weight_decay=arch["train"]["w_decay"])
        self.scheduler = build_scheduler(self.optimizer, arch["train"])
        self.scaler = torch.amp.GradScaler("cuda")
        if {id(p) for g in self.optimizer.param_groups for p in g["params"]} != {id(p) for p in self.model.parameters()}:
            raise ValueError("Optimizer must contain exactly one shared model")
        self.sampler = d4.EffectiveBatchSampler(len(self.parser.train_dataset), arch["train"]["micro_batch_size"], 0)
        self.train_generator = torch.Generator()
        kwargs = dict(batch_sampler=self.sampler, num_workers=arch["train"]["workers"],
                      pin_memory=True, collate_fn=bev_collate_fn, generator=self.train_generator)
        if kwargs["num_workers"]:
            kwargs.update(persistent_workers=False, prefetch_factor=4)
        self.train_loader = DataLoader(self.parser.train_dataset, **kwargs)
        from evaluate_pointnet_class_fixed_alpha import capture_protected
        protected = capture_protected(ROOT/"logs/SJNet3_PointNet_stepD1b_high_alpha_sensitivity")
        for folder in (source, ROOT/"logs/SJNet3_PointNet_stepD4_soft_teacher_student_cp0_lr1e4"):
            protected.update({str(p.resolve()):d1.file_sha256(p) for p in folder.rglob("*") if p.is_file()})
        for p in (ROOT/"train_pointnet_shared_soft_alpha_d5.py", ROOT/"evaluate_pointnet_class_fixed_alpha.py",
                  ROOT/"logs/SJNet3_PointNet_stepC2_local3_gated_lr2e3_bs3/best_val.path"):
            protected[str(p)] = d1.file_sha256(p)
        self.metadata = d1b.json_normalized(dict(experiment="Step D.5", arch_cfg=arch, data_cfg=data,
            initialization="C.2 seed-0 random initialization; segmentation network only, fixed pretrained PointNet features",
            initial_model_sha256=self.initial_hash, c2_random_initialization_sha256=c2_random_hash,
            c2_trained_ema_sha256=ref["model_state_sha256"], pretrained_initialization=False,
            disposable_preflight=disposable, parameters=28876586, teacher_exists=False,
            alpha_table=list(ALPHA_TABLE), temperature=1., stage1_alpha=1.2, unobserved_alpha=1.2,
            input_features="7 handcrafted + 16 fixed PointNet + 1 mask", copy_paste_prob=0.,
            dataset_root=str(Path(dataset).resolve()), validation_manifest_sha256=manifest_hash,
            train_scans=len(self.parser.train_dataset), valid_scans=4071, valid_batch_size=3,
            effective_batch_size=3, micro_batch_size=arch["train"]["micro_batch_size"],
            training_metric="C.2 sample-weighted average of cumulative confusion metrics at EFFECTIVE batch boundaries",
            precision="C.2 BF16 autocast + GradScaler for Stage 2 training; Stage 1/alpha and all validation FP32",
            ema_decay=.99, normalization_batchnorm_modules=sum(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in self.model.modules()),
            accumulation_caveat="Sample-size weighting 2/3+1/3 (or thirds); weighted Focal/Lovasz reductions and dropout need not equal an unsplit batch",
            aux_policy="C.2 aux2=.30, aux4=.15; linearly decay epochs 50..60, then zero; aux8=.10 unchanged",
            training_worker_policy="seeded fresh workers each epoch for epoch-boundary resume",
            protected_files=protected, torch_version=str(torch.__version__), gpu=torch.cuda.get_device_name(),
            cuda_allocator_memory_fraction=.85, cudnn_deterministic=True, cudnn_benchmark=True,
            independent_test=False, exploratory_development_evaluation=True,
            c2_trainer_reference_commit="396385a5051d3358b9d4c471268ca8be4e09ff3d"))
        self.metadata_id = hashlib.sha256(json.dumps(self.metadata, sort_keys=True).encode()).hexdigest()
        self.early_stopping = EarlyStopping(**arch["train"]["early_stopping"])
        self.best_val_iou, self.best_val_epoch, self.next_epoch = -1., -1, 0
        self.initialization_verified, self.history, self.total_train_seconds = False, [], 0.
        if resume:
            saved = torch.load(resume, map_location="cpu", weights_only=False)
            validate_resume(saved, arch, data, self.metadata_id)
            self.model.load_state_dict(saved["model_train"], strict=True)
            self.ema_model.load_state_dict(saved["model_ema"], strict=True)
            self.optimizer.load_state_dict(saved["optim"])
            self.scheduler.load_state_dict(saved["scheduler"])
            self.scaler.load_state_dict(saved["scaler"])
            self.early_stopping.load_state_dict(saved["early_stopping"])
            self.next_epoch = saved["epoch"]+1
            self.best_val_iou, self.best_val_epoch = saved["best_val_iou"], saved["best_val_epoch"]
            self.history, self.total_train_seconds = saved["history"], saved["total_train_seconds"]
            self.initialization_verified = True
            d4.restore_rng(saved["rng_state"])
            self.logger.info(f"Resuming D.5 | saved epoch={saved['epoch']} | next epoch={self.next_epoch} | Best Validation mIoU={self.best_val_iou:.9f} | Best Epoch={self.best_val_epoch}; optimizer/scheduler/scaler/EMA/RNG restored")

    def set_epoch_loss_weights(self, epoch):
        ratio = 1. if epoch < 50 else max(0., (60-epoch)/10.)
        self.w_aux2, self.w_aux4 = .30*ratio, .15*ratio

    def batch_input(self, batch):
        return d4.Trainer.batch_input(self, batch)

    def full_loss(self, outputs, labels, mask):
        return d4.Trainer.full_loss(self, outputs, labels, mask)

    def evaluator(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return d1.iouEval(20, self.device, [0])

    @torch.no_grad()
    def initial_check(self):
        self.model.eval()
        x, _, mask = self.batch_input(next(iter(self.parser.validloader)))
        logits, alpha = self.model.stage1(x)
        outputs = self.model.forward_with_alpha(x, alpha_map=alpha)
        assert not logits.requires_grad and not alpha.requires_grad and not torch.is_inference(alpha)
        assert torch.isfinite(alpha).all() and alpha.min() >= .6-1e-6 and alpha.max() <= 1.6+1e-6
        torch.testing.assert_close(alpha[mask<=0], torch.full_like(alpha[mask<=0], 1.2), atol=0, rtol=0)
        torch.testing.assert_close(logits.softmax(1).sum(1), torch.ones_like(logits[:, 0]), atol=1e-6, rtol=1e-6)
        assert set(outputs) == {"logits", "aux2", "aux4", "aux8", "boundary"}
        for name, tensor in outputs.items():
            assert tuple(tensor.shape) == (3, 1 if name == "boundary" else 20, 512, 512)
            assert torch.isfinite(tensor).all()
        assert d1.state_digest(self.model.state_dict()) == self.initial_hash
        assert d1.state_digest(self.ema_model.state_dict()) == self.initial_hash
        self.initialization_verified = True
        return dict(status="PASS", scans=3, initialization="random", initial_state_sha256=self.initial_hash,
                    differs_from_C2_trained=True, matches_C2_random=True, teacher_exists=False,
                    stage1_gradient=False, alpha_gradient=False, alpha_min=alpha.min().item(), alpha_max=alpha.max().item(),
                    output_shapes={key:list(value.shape) for key, value in outputs.items()})

    def train_epoch(self, epoch, max_updates=None, report=True):
        self.model.train()
        self.set_epoch_loss_weights(epoch)
        self.sampler.epoch = epoch
        self.train_generator.manual_seed(epoch)
        evaluator = self.evaluator()
        losses, accuracy, iou = AverageMeter(), AverageMeter(), AverageMeter()
        total_batches = len(self.parser.train_dataset)//3
        accumulated, group_loss, updates = 0, 0., 0
        started = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        for batch in self.train_loader:
            x, labels, mask = self.batch_input(batch)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = self.model(x)
                loss = self.full_loss(outputs, labels, mask)
            size = len(x)
            self.scaler.scale(loss*(size/3.)).backward()
            accumulated += size
            group_loss += loss.item()*size/3.
            with torch.no_grad():
                predictions = d1.original_predictions(outputs["logits"], x[:, :23])
                evaluator.addBatch(predictions, labels)
            if accumulated == 3:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self._update_ema()
                self.optimizer.zero_grad(set_to_none=True)
                losses.update(group_loss, 3)
                accuracy.update(evaluator.getacc().item(), 3)
                iou.update(evaluator.getIoU()[0].item(), 3)
                if report and should_report(updates, self.ARCH["train"]["report_batch"]):
                    log_training_progress(self.logger, epoch, updates, total_batches,
                                          self.optimizer.param_groups[0]["lr"], losses, accuracy, iou)
                updates += 1
                accumulated, group_loss = 0, 0.
            del outputs, loss, predictions, x, labels, mask
            if max_updates is not None and updates >= max_updates:
                break
        if accumulated:
            raise RuntimeError("Incomplete effective batch at epoch boundary")
        return dict(loss=losses.avg, accuracy=accuracy.avg, miou=iou.avg, optimizer_updates=updates,
                    samples=updates*3, seconds=time.perf_counter()-started,
                    final_global_miou=evaluator.getIoU()[0].item())

    def smoke(self):
        if not self.metadata["disposable_preflight"]:
            raise RuntimeError("Smoke updates may only run on a disposable model")
        torch.cuda.reset_peak_memory_stats()
        initial = self.initial_check()
        before = self.initial_hash
        trained = self.train_epoch(0, max_updates=3, report=False)
        assert d1.state_digest(self.model.state_dict()) != before
        assert d1.state_digest(self.ema_model.state_dict()) != before
        assert all(p.grad is None for p in self.ema_model.parameters())
        path = self.output/"resume_smoke.path"
        d4.atomic_checkpoint(path, self.checkpoint(-1))
        return dict(status="PASS", initialization=initial, training=trained,
                    checkpoint=str(path), micro_batch_size=self.ARCH["train"]["micro_batch_size"], effective_batch_size=3,
                    peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                    peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
                    shared_weights=True, teacher_exists=False, ema_updated=True)

    def resume_smoke(self, checkpoint):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        for key, current in (("model_train", self.model.state_dict()), ("model_ema", self.ema_model.state_dict()),
                             ("optim", self.optimizer.state_dict()), ("scheduler", self.scheduler.state_dict()),
                             ("scaler", self.scaler.state_dict()), ("early_stopping", self.early_stopping.state_dict()),
                             ("rng_state", d4.rng_state())):
            if nested_digest(saved[key]) != nested_digest(current):
                raise RuntimeError(f"D.5 actual checkpoint resume mismatch: {key}")
        before = d1.state_digest(self.model.state_dict())
        metrics = self.train_epoch(0, max_updates=1, report=False)
        assert before != d1.state_digest(self.model.state_dict())
        return dict(status="PASS", restored=["model_train", "model_ema", "optim", "scheduler", "scaler", "early_stopping", "rng_state"],
                    additional_optimizer_updates=metrics["optimizer_updates"], disposable_weights_discarded=True)

    @torch.no_grad()
    def validate(self, epoch, tag=None):
        self.set_epoch_loss_weights(epoch)
        target = self.ema_model.eval()
        first, second = self.evaluator(), self.evaluator()
        stats, raw_histogram = AlphaStatistics(), np.zeros(20, dtype=np.int64)
        samples, total_loss = 0, 0.
        started = time.perf_counter()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        forward_seconds = 0.
        for index, batch in enumerate(self.parser.validloader):
            ids = [f"{seq}/{name}" for seq, name in zip(batch[4], batch[5])]
            if ids != self.scan_ids[samples:samples+len(ids)]:
                raise RuntimeError("Validation scan order changed")
            x, labels, mask = self.batch_input(batch)
            begin.record()
            logits1, alpha = target.stage1(x)
            outputs = target.forward_with_alpha(x, alpha_map=alpha)
            end.record()
            end.synchronize()
            forward_seconds += begin.elapsed_time(end)/1000.
            loss = self.full_loss(outputs, labels, mask)
            observed = mask[:, 0] > 0
            raw = logits1.argmax(1)[observed].cpu().numpy()
            raw_histogram += np.bincount(raw, minlength=20)
            for evaluator, logits in ((first, logits1), (second, outputs["logits"])):
                evaluator.addBatch(d1.original_predictions(logits, x[:, :23]), labels)
            values = alpha[:, 0][observed].cpu().numpy()
            groups = np.zeros(len(values), dtype=np.int64)
            # Reuse D.2's population statistics; mode 0 supplies cell counts.
            for mode in (0, 1):
                stats._add(mode, values, groups, 0, 1)
            stats.unobserved += int(observed.numel()-observed.count_nonzero().item())
            total_loss += loss.item()*len(ids)
            samples += len(ids)
            del x, labels, mask, logits1, alpha, outputs, loss, observed
            if (index+1) % 100 == 0:
                d1.atomic_json(self.output/"validation_progress.json", dict(epoch=epoch, scans=samples, total=4071))
        if samples != 4071:
            raise RuntimeError("D.5 requires the full 4071-scan validation")
        result = dict(epoch=epoch, scans=samples, seconds=time.perf_counter()-started,
                      forward_cuda_seconds=forward_seconds, weights="EMA", alpha_statistics={
                          scope:stats.distribution(1, scope=scope) for scope in ("observed", "unobserved", "all")},
                      stage1_raw_prediction_histogram_observed=raw_histogram.tolist())
        for name, evaluator in (("stage1", first), ("stage2", second)):
            miou, per_class = evaluator.getIoU()
            result[name] = dict(miou=miou.item(), accuracy=evaluator.getacc().item(),
                per_class_iou={CLASS_NAMES[cid]:per_class[cid].item() for cid in range(1, 20)},
                confusion_matrix=evaluator.conf_matrix.cpu().tolist())
        result["stage2"]["loss"] = total_loss/samples
        result["stage2_minus_stage1"] = result["stage2"]["miou"]-result["stage1"]["miou"]
        name = tag or f"validation_epoch_{epoch:02d}"
        d1.atomic_json(self.output/f"{name}.json", result)
        histogram = stats.hist[1, 0]
        d1.write_csv(self.output/f"{name}_alpha_histogram.csv", ["lower", "upper", "observed_count"],
            [dict(lower=.6+i*HIST_WIDTH, upper=min(1.6, .6+(i+1)*HIST_WIDTH), observed_count=int(histogram[i]))
             for i in np.flatnonzero(histogram)])
        return result

    def checkpoint(self, epoch, is_best=False):
        raw, ema = self.model.state_dict(), self.ema_model.state_dict()
        return dict(experiment="Step D.5", model_name=self.ARCH["model"]["name"], epoch=epoch,
            model=ema if is_best else raw, model_train=raw, model_ema=ema,
            optim=self.optimizer.state_dict(), scheduler=self.scheduler.state_dict(), scaler=self.scaler.state_dict(),
            early_stopping=self.early_stopping.state_dict(), best_val_iou=self.best_val_iou, best_val_epoch=self.best_val_epoch,
            alpha_table=list(ALPHA_TABLE), copy_paste_prob=0., pretrained_initialization=False, seed=0,
            initialization_verified=self.initialization_verified, initial_state_sha256=self.initial_hash,
            arch_cfg=self.ARCH, data_cfg=self.DATA, metadata=self.metadata, metadata_id=self.metadata_id,
            history=self.history, total_train_seconds=self.total_train_seconds, rng_state=d4.rng_state())

    def prepare(self):
        if self.initialization_verified:
            return
        report = self.initial_check()
        d1.atomic_json(self.output/"random_initialization_check.json", report)
        state = self.checkpoint(-1)
        for name in ("epoch0_random_init.path", "latest.path"):
            d4.atomic_checkpoint(self.output/name, state)
        self.logger.info(f"Random initialization verified | state hash: {self.initial_hash} | C.2-trained weights NOT loaded | saved epoch0_random_init.path (before training epoch 0)")

    def write_history(self):
        fields = ["epoch", "lr", "train_loss", "train_accuracy", "train_miou", "stage1_miou", "stage1_accuracy",
                  "stage2_miou", "stage2_accuracy", "stage2_minus_stage1", "stage2_loss", "best_miou", "best_epoch",
                  "early_stopping_counter", "seconds", "peak_allocated_GiB", "peak_reserved_GiB"]
        d1.write_csv(self.output/"validation_history.csv", fields, [{key:row[key] for key in fields} for row in self.history])
        d1.atomic_json(self.output/"training_history.json", self.history)
        rows = [dict(epoch=row["epoch"], scope=scope, **values)
                for row in self.history for scope, values in row["validation"]["alpha_statistics"].items()]
        if rows:
            d1.write_csv(self.output/"alpha_statistics.csv", list(rows[0]), rows)

    def train(self):
        if not self.initialization_verified:
            raise RuntimeError("Formal training requires the random initialization check")
        if self.early_stopping.should_stop(self.next_epoch-1):
            self.logger.info("Checkpoint already reached early stopping; no further training.")
            return
        for epoch in range(self.next_epoch, self.ARCH["train"]["max_epochs"]):
            started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            rate = self.optimizer.param_groups[0]["lr"]
            self.logger.info(f"Epoch {epoch:02d} | Learning rate: {rate:.6e}")
            train = self.train_epoch(epoch)
            self.scheduler.step()
            valid = self.validate(epoch)
            stop = self.early_stopping.update(valid["stage2"]["miou"], epoch)
            improved = valid["stage2"]["miou"] > self.best_val_iou
            if improved:
                self.best_val_iou, self.best_val_epoch = valid["stage2"]["miou"], epoch
            elapsed = time.perf_counter()-started
            self.total_train_seconds += elapsed
            row = dict(epoch=epoch, lr=rate, train_loss=train["loss"], train_accuracy=train["accuracy"], train_miou=train["miou"],
                stage1_miou=valid["stage1"]["miou"], stage1_accuracy=valid["stage1"]["accuracy"],
                stage2_miou=valid["stage2"]["miou"], stage2_accuracy=valid["stage2"]["accuracy"],
                stage2_minus_stage1=valid["stage2_minus_stage1"], stage2_loss=valid["stage2"]["loss"],
                best_miou=self.best_val_iou, best_epoch=self.best_val_epoch, early_stopping_counter=self.early_stopping.bad_epochs,
                seconds=elapsed, peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30, validation=valid,
                model_train_sha256=d1.state_digest(self.model.state_dict()), model_ema_sha256=d1.state_digest(self.ema_model.state_dict()))
            self.history.append(row)
            self.write_history()
            log_validation_summary(self.logger, epoch, rate, valid, self.best_val_iou, self.best_val_epoch, self.early_stopping)
            if improved:
                self.logger.info("Best mean IoU on validation so far, saving!")
                d4.atomic_checkpoint(self.output/"best_val.path", self.checkpoint(epoch, is_best=True))
                d1.atomic_json(self.output/"best_validation.json", valid)
                self.logger.info(f"Saved checkpoint: {self.output/'best_val.path'} (epoch {epoch}, Stage 2 EMA mIoU {self.best_val_iou:.9f}; both stages' 19 class IoUs saved)")
            d4.atomic_checkpoint(self.output/"latest.path", self.checkpoint(epoch))
            self.logger.info(f"Saved checkpoint: {self.output/'latest.path'} (epoch {epoch}) | seconds: {elapsed:.1f} | peak allocated/reserved GiB: {row['peak_allocated_GiB']:.3f}/{row['peak_reserved_GiB']:.3f}")
            d1.atomic_json(self.output/"status.json", dict(status="early_stopped" if stop else "training", epoch=epoch,
                best_epoch=self.best_val_epoch, best_miou=self.best_val_iou))
            if stop:
                self.logger.info(f"Early stopping triggered at epoch {epoch}: bad epochs {self.early_stopping.bad_epochs}/{self.early_stopping.patience}, min_epochs={self.early_stopping.min_epochs}, significant min_delta={self.early_stopping.min_delta}; Best epoch: {self.best_val_epoch}; Best validation mIoU: {self.best_val_iou:.9f}")
                break
            if epoch < self.early_stopping.min_epochs and self.early_stopping.bad_epochs >= self.early_stopping.patience:
                self.logger.info(f"Early stopping deferred: epoch {epoch} < min_epochs {self.early_stopping.min_epochs}; continuing.")

    def finalize(self):
        saved = torch.load(self.output/"best_val.path", map_location="cpu", weights_only=False)
        validate_resume(saved, self.ARCH, self.DATA, self.metadata_id)
        self.ema_model.load_state_dict(saved["model_ema"], strict=True)
        best = self.validate(saved["best_val_epoch"], tag="best_ema_validation")
        if abs(best["stage2"]["miou"]-saved["best_val_iou"]) > .0001:
            raise RuntimeError("Best EMA re-evaluation differs from its saved score")
        rows = [dict(Class=name, D2_Soft=self.reference["per_class_iou"][name],
                     D5_Stage1=best["stage1"]["per_class_iou"][name], D5_Stage2=best["stage2"]["per_class_iou"][name],
                     delta_pp=100*(best["stage2"]["per_class_iou"][name]-self.reference["per_class_iou"][name])) for name in CLASS_NAMES[1:]]
        d1.write_csv(self.output/"best_class_comparison.csv", list(rows[0]), rows)
        d4_result = json.loads((ROOT/"logs/SJNet3_PointNet_stepD4_soft_teacher_student_cp0_lr1e4/results.json").read_text())
        comparisons = dict(Baseline_8ch=.507, C2=.508252935, D1b=self.reference_run["reference"]["miou"],
            D2b=self.reference["miou"], D4=d4_result["best_miou"],
            D5_Stage1=best["stage1"]["miou"], D5_Stage2=best["stage2"]["miou"])
        result = dict(status="complete", best_epoch=saved["best_val_epoch"], best=best, comparisons=comparisons,
            delta_vs_D2_pp=100*(best["stage2"]["miou"]-self.reference["miou"]),
            improved_classes=[row["Class"] for row in rows if row["delta_pp"] > 0],
            worsened_classes=[row["Class"] for row in rows if row["delta_pp"] < 0],
            motorcyclist_above_zero=best["stage2"]["per_class_iou"]["motorcyclist"] > 0,
            early_stopping=self.early_stopping.state_dict(), training_validation_seconds=self.total_train_seconds,
            initialization_sha256=self.initial_hash, metadata_id=self.metadata_id)
        report = ["# Step D.5 Shared-weight Two-pass Soft Alpha Scratch", "",
            "Exploratory sequence-08 development evaluation, not an independent test. No alpha retuning.",
            "One shared trainable model plus its EMA; no Teacher. Segmentation weights were random; fixed precomputed PointNet features are reused.",
            f"Best epoch: {saved['best_val_epoch']}; Stage 2 EMA mIoU: {best['stage2']['miou']:.12f}; D.2b difference: {result['delta_vs_D2_pp']:+.6f} pp.",
            "", "| Method | mIoU |", "|---|---|"]
        report += [f"| {name} | {score:.12f} |" for name, score in comparisons.items()]
        report += ["", "| Class | D.2b Soft | D.5 Stage 1 | D.5 Stage 2 | Delta vs D.2b (pp) |", "|---|---|---|---|---|"]
        report += [f"| {r['Class']} | {r['D2_Soft']:.9f} | {r['D5_Stage1']:.9f} | {r['D5_Stage2']:.9f} | {r['delta_pp']:+.6f} |" for r in rows]
        report += ["", f"Improved classes: {len(result['improved_classes'])}; worsened: {len(result['worsened_classes'])}.",
            f"Motorcyclist above zero: {result['motorcyclist_above_zero']}. Stage 2 minus Stage 1: {best['stage2_minus_stage1']:+.9f}.",
            f"Training + per-epoch validation: {self.total_train_seconds:.1f} seconds; best evaluation: {best['seconds']:.1f} seconds.",
            f"Early stopping: {self.early_stopping.state_dict()}.",
            "C.2 BF16 Stage 2 training, FP32 Stage 1/validation; original aux loss decay epochs 50..60 retained. CP=0.",
            self.metadata["accumulation_caveat"],
            "A matched scratch global-alpha=1.2 control is still needed to isolate causal Soft-alpha gains."]
        (self.output/"comparison.md").write_text("\n".join(report)+"\n")
        d4.atomic_checkpoint(self.output/"inference_package.path", dict(model=self.ema_model.state_dict(),
            arch_cfg=self.ARCH, data_cfg=self.DATA, alpha_table=list(ALPHA_TABLE), metadata=self.metadata,
            best_epoch=saved["best_val_epoch"], stage2_miou=best["stage2"]["miou"]))
        d1b.verify_protected(self.metadata["protected_files"])
        d1.atomic_json(self.output/"results.json", result)
        d1.atomic_json(self.output/"status.json", dict(status="complete", best_epoch=saved["best_val_epoch"], best_miou=best["stage2"]["miou"]))
        self.logger.info(f"Finished Training | Best Epoch: {saved['best_val_epoch']} | Best Stage 2 Validation mIoU: {best['stage2']['miou']:.9f} | comparison.md saved")
        return result
