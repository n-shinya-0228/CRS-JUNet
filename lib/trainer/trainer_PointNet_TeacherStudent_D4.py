"""D.4 Teacher-conditioned Student Fine-tuning; original C.2 loss/EMA methods."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler

import evaluate_pointnet_fixed_alpha_sensitivity as d1
import evaluate_pointnet_high_alpha_sensitivity as d1b
from evaluate_pointnet_class_fixed_alpha import validate_alpha_table
from lib.dataset.Parser_PointNet import Parser, bev_collate_fn
from lib.losses import Lovasz_softmax
from lib.models.SJNet3_PointNet_Gated_Local3_ClassAlpha import ALPHA_TABLE, CLASS_NAMES, build_alpha_maps
from lib.models.SJNet3_PointNet_Gated_Local3_TeacherStudent import SJNet3_PointNet_Gated_Local3_TeacherStudent
from lib.utils.pointnet_class_alpha_statistics import AlphaStatistics
from lib.utils.pointnet_d4_training import EarlyStopping, build_scheduler

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("d4_readonly_pointnet_trainer", Path(__file__).with_name("trainer_PointNet.py"))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
C2LossEMATrainer = _base.Trainer
EXPECTED_D2 = .5097046791309


class EffectiveBatchSampler(Sampler):
    """Keep groups of three; split only for memory, never add extra examples."""
    def __init__(self, size, micro_batch_size=3, seed=0):
        if micro_batch_size not in (1, 2, 3):
            raise ValueError("micro_batch_size must be 1, 2 or 3")
        self.size, self.micro, self.seed, self.epoch = size, micro_batch_size, seed, 0

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed+self.epoch)
        indices = torch.randperm(self.size, generator=generator).tolist()
        for offset in range(0, self.size-self.size % 3, 3):
            group = indices[offset:offset+3]
            for start in range(0, 3, self.micro):
                yield group[start:start+self.micro]

    def __len__(self):
        return self.size//3 * ((3+self.micro-1)//self.micro)


def validate_config(arch):
    train = arch["train"]
    if arch["model"]["name"] != "SJNet3_PointNet_Gated_Local3_TeacherStudent":
        raise ValueError("D.4 requires its own TeacherStudent config")
    if (train["batch_size"] != 3 or train.get("copy_paste_prob") != 0 or train.get("amp", False)
            or not train.get("use_ema", True) or train.get("seed", 0) != 0):
        raise ValueError("D.4 requires effective batch 3, CP=0, FP32, Student EMA and seed 0")
    if train["max_epochs"] != 20 or train["wup_epochs"] != 1 or train["cos_epochs"] != 19:
        raise ValueError("D.4 uses 20 epochs with 1 warmup + 19 cosine")
    if train.get("micro_batch_size", 3) not in (1, 2, 3) or train["report_epoch"] != 1:
        raise ValueError("Invalid D.4 microbatch or validation frequency")
    if not 0 < train["min_lr"] < train["lr"]:
        raise ValueError("Require 0 < min_lr < base lr")


def validate_resume(saved, arch, data, identity, teacher):
    if (saved.get("experiment") != "Step D.4" or saved.get("arch_cfg") != arch or
            saved.get("data_cfg") != data or saved.get("metadata_id") != identity or
            saved.get("teacher_identity") != teacher or saved.get("alpha_table") != list(ALPHA_TABLE) or
            saved.get("copy_paste_prob") != 0 or not saved.get("reproduction_gate_passed")):
        raise ValueError("Resume must match D.4 config, code/data identity, fixed teacher/table and passed gate")


def atomic_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(saved):
    random.setstate(saved["python"])
    np.random.set_state(saved["numpy"])
    torch.set_rng_state(saved["torch"].cpu())
    if saved["cuda"] is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in saved["cuda"]])


class Trainer(C2LossEMATrainer):
    def __init__(self, arch, data, dataset, output, logger, teacher_checkpoint, resume=None):
        validate_config(arch)
        self.ARCH, self.DATA, self.log, self.logger = arch, data, str(output), logger
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            raise RuntimeError("D.4 requires CUDA for real-data execution")
        # Leave room for the Windows display/driver instead of GPU host paging.
        torch.cuda.set_per_process_memory_fraction(.85, self.device)
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.fastest = True
        source = ROOT/"logs/SJNet3_PointNet_stepD2_class_fixed_alpha"
        self.reference_run = json.loads((source/"results.json").read_text())
        if self.reference_run["status"] != "complete":
            raise ValueError("D.2 must have completed validation")
        self.reference = self.reference_run["results"]["D2_Soft"]
        if abs(self.reference["miou"]-EXPECTED_D2) > 1e-12:
            raise ValueError("Unexpected stored D.2b reference; do not retune the experiment")
        teacher_checkpoint = Path(teacher_checkpoint).resolve()
        checksum = d1.file_sha256(teacher_checkpoint)
        reference_metadata = self.reference_run["metadata"]
        if teacher_checkpoint.name != "best_val.path" or checksum != reference_metadata["checkpoint_sha256"]:
            raise ValueError("Require the same C.2 best checkpoint as D.2")
        checkpoint = torch.load(teacher_checkpoint, map_location="cpu", weights_only=False)
        base_arch = checkpoint["arch_cfg"]
        state, weights = d1.select_evaluation_state(checkpoint, base_arch, data)
        if arch["dataset"] != base_arch["dataset"] or arch["model"]["dropout"] != base_arch["model"]["dropout"]:
            raise ValueError("D.4 must preserve the C.2 input and dropout settings")
        self.teacher_identity = dict(path=str(teacher_checkpoint), sha256=checksum,
                                     state_sha256=d1.state_digest(state), epoch=18, alpha=1.2, weights=weights)
        self.c2_best_miou = float(checkpoint["best_val_iou"])
        if self.teacher_identity["state_sha256"] != reference_metadata["model_state_sha256"]:
            raise ValueError("D.4 EMA initialization differs from D.2")
        self.parser = Parser(root=dataset, data_cfg=data, arch_cfg=arch, gt=True, shuffle_train=False)
        classes = {cid:self.parser.get_xentropy_class_string(cid) for cid in range(20)}
        validate_alpha_table(ROOT/"logs/SJNet3_PointNet_stepD1b_high_alpha_sensitivity", classes)
        if (self.parser.train_dataset.copy_paste_prob != 0 or self.parser.valid_dataset.copy_paste_prob != 0
                or not self.parser.train_dataset.is_train or self.parser.valid_dataset.is_train
                or data["split"]["valid"] != [8] or self.parser.validloader.batch_size != 3):
            raise ValueError("Require original train augmentation, CP OFF and unaugmented validation batch 3")
        entries, self.scan_ids, manifest_hash = d1.dataset_manifest(self.parser.valid_dataset, dataset)
        if (len(self.scan_ids) != 4071 or manifest_hash != reference_metadata["validation_manifest_sha256"]
                or str(Path(dataset).resolve()) != reference_metadata["dataset_root"]
                or dict(entries=entries, scan_ids=self.scan_ids) != json.loads((source/"validation_manifest.json").read_text())):
            raise ValueError("D.4 validation files/order differ from D.2")
        if (d1b.json_normalized(data) != reference_metadata["data_cfg"] or
                str(torch.__version__) != reference_metadata["torch_version"] or
                torch.cuda.get_device_name() != reference_metadata["gpu"]):
            raise ValueError("D.4 labels or precision environment differs from D.2")
        self.validation_entries = entries
        self.system = SJNet3_PointNet_Gated_Local3_TeacherStudent(
            in_channels=24, nclasses=20, drop=arch["model"]["dropout"])
        self.system.initialize(state)
        self.model = self.system.student
        for name, model in (("teacher", self.system.teacher), ("student", self.model)):
            if sum(p.numel() for p in model.parameters()) != 28876586 or d1.state_digest(model.state_dict()) != self.teacher_identity["state_sha256"]:
                raise ValueError(name+" does not strictly match the C.2 EMA state")
        self.system.to(self.device)
        self.use_ema, self.ema_decay = True, float(arch["train"]["ema_decay"])
        self._build_ema_model()
        self.ema_model.eval()
        content = torch.zeros(20, dtype=torch.float32)
        for cid, frequency in data["content"].items():
            content[self.parser.to_xentropy(cid)] += frequency
        probability = content/(content.sum()+1e-6)
        self.loss_w = torch.clamp(1/(probability+arch["train"]["epsilon_w"]), max=20.)
        for cid, ignored in data["learning_ignore"].items():
            if ignored:
                self.loss_w[cid] = 0
        self.criterion_main = _base.FocalLoss(weight=self.loss_w, gamma=1., ignore_index=0).to(self.device)
        self.lovasz = Lovasz_softmax(ignore=0).to(self.device)
        self.boundary_criterion = nn.BCEWithLogitsLoss(reduction="none").to(self.device)
        self.sobel = _base.SobelFilter().to(self.device)
        self.w_aux2, self.w_aux4, self.w_aux8 = .30, .15, .10
        self.w_lovasz, self.w_boundary, self.w_paste = .50, .20, 0.
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=arch["train"]["lr"], weight_decay=arch["train"]["w_decay"])
        self.scheduler = build_scheduler(self.optimizer, arch["train"])
        optimizer_ids = {id(p) for group in self.optimizer.param_groups for p in group["params"]}
        if optimizer_ids != {id(p) for p in self.model.parameters()} or optimizer_ids & {id(p) for p in self.system.teacher.parameters()}:
            raise ValueError("Only Student parameters may enter AdamW")
        self.sampler = EffectiveBatchSampler(len(self.parser.train_dataset), arch["train"]["micro_batch_size"], 0)
        self.train_generator = torch.Generator()
        kwargs = dict(batch_sampler=self.sampler, num_workers=arch["train"]["workers"],
                      pin_memory=True, collate_fn=bev_collate_fn, generator=self.train_generator)
        if kwargs["num_workers"]:
            kwargs.update(persistent_workers=False, prefetch_factor=4)
        self.train_loader = DataLoader(self.parser.train_dataset, **kwargs)
        from evaluate_pointnet_class_fixed_alpha import capture_protected
        protected = capture_protected(ROOT/"logs/SJNet3_PointNet_stepD1b_high_alpha_sensitivity")
        protected.update({str(p.resolve()):d1.file_sha256(p) for p in source.rglob("*") if p.is_file()})
        protected[str(teacher_checkpoint)] = checksum
        protected[str(ROOT/"train_pointnet_teacher_student_d4.py")] = d1.file_sha256(ROOT/"train_pointnet_teacher_student_d4.py")
        self.metadata = d1b.json_normalized(dict(experiment="Step D.4", description="Teacher-conditioned Student Fine-tuning, not knowledge distillation",
            arch_cfg=arch, data_cfg=data, teacher=self.teacher_identity, source_d2_results_sha256=d1.file_sha256(source/"results.json"),
            alpha_table=list(ALPHA_TABLE), input_features="7 handcrafted + 16 fixed PointNet + observation mask",
            validation_manifest_sha256=manifest_hash, dataset_root=str(Path(dataset).resolve()),
            train_scans=len(self.parser.train_dataset), valid_scans=4071, valid_batch_size=3,
            effective_train_batch_size=3, micro_batch_size=arch["train"]["micro_batch_size"],
            training_worker_policy="fresh workers per epoch and seeded generator; exact epoch-boundary resume",
            training_loss_metric="sample-weighted mean of cumulative microbatch confusion metrics, as C.2 Trainer; final global metrics also saved",
            normalization_batchnorm_modules=sum(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in self.model.modules()),
            loss_weights=dict(main=1., aux2=.30, aux4=.15, aux8=.10, lovasz=.50, boundary=.20, paste=0.),
            loss_main="C.2 content-normalized weighted Focal gamma=1, ignore=0", ema_decay=self.ema_decay,
            initial_teacher_state_sha256=self.teacher_identity["state_sha256"],
            initial_student_state_sha256=self.teacher_identity["state_sha256"],
            initial_student_ema_state_sha256=self.teacher_identity["state_sha256"],
            precision="float32, no autocast/scaler; C.2 training used BF16, D.2 validation used float32",
            protected_files=protected, torch_version=str(torch.__version__), gpu=torch.cuda.get_device_name(),
            cuda_allocator_memory_fraction=.85,
            cudnn_deterministic=True, cudnn_benchmark=False,
            cudnn_policy="disable timing-based autotuning for stable frozen Teacher maps across native resume; D.2 used benchmark=True",
            exploratory_development_evaluation=True, independent_test=False, seed=0,
            c2_trainer_reference_commit="396385a5051d3358b9d4c471268ca8be4e09ff3d",
            reused_C2_methods=["_mix_losses", "_compute_boundary_gt", "_build_ema_model", "_update_ema"]))
        self.metadata_id = hashlib.sha256(json.dumps(self.metadata, sort_keys=True).encode()).hexdigest()
        self.early_stopping = EarlyStopping(**arch["train"]["early_stopping"])
        self.best_val_iou, self.best_val_epoch, self.next_epoch = -1., 0, 1
        self.gate_passed = False
        self.history = []
        self.total_train_seconds = 0.
        if resume:
            saved = torch.load(resume, map_location="cpu", weights_only=False)
            validate_resume(saved, arch, data, self.metadata_id, self.teacher_identity)
            self.model.load_state_dict(saved["model_train"], strict=True)
            self.ema_model.load_state_dict(saved["model_ema"], strict=True)
            self.optimizer.load_state_dict(saved["optim"])
            for values in self.optimizer.state.values():
                for key, value in values.items():
                    if isinstance(value, torch.Tensor):
                        values[key] = value.to(self.device)
            self.scheduler.load_state_dict(saved["scheduler"])
            self.early_stopping.load_state_dict(saved["early_stopping"])
            self.next_epoch, self.best_val_iou, self.best_val_epoch = saved["epoch"]+1, saved["best_val_iou"], saved["best_val_epoch"]
            self.history, self.total_train_seconds = saved["history"], saved["total_train_seconds"]
            self.gate_passed = True
            restore_rng(saved["rng_state"])
        self.assert_teacher_fixed()
        del state, checkpoint

    def assert_teacher_fixed(self):
        teacher = self.system.teacher
        if teacher.training or any(p.requires_grad or p.grad is not None for p in teacher.parameters()):
            raise RuntimeError("Teacher is not fully frozen/eval")
        if d1.state_digest(teacher.state_dict()) != self.teacher_identity["state_sha256"]:
            raise RuntimeError("Teacher state hash changed")

    def batch_input(self, batch):
        if torch.count_nonzero(batch[14]).item():
            raise RuntimeError("Copy-Paste OFF requires an all-zero paste mask")
        features = batch[0].to(self.device)
        mask = batch[1].to(self.device).float()
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        x = torch.cat((features, mask), dim=1)
        if x.shape[1:] != (24, 512, 512) or x.dtype != torch.float32:
            raise ValueError("Expected same float32 [B,24,512,512] input for both networks")
        return x, batch[2].to(self.device).long(), mask

    def full_loss(self, outputs, labels, mask):
        boundary = self._compute_boundary_gt(labels)*mask
        loss = self._mix_losses(outputs, labels, boundary, mask, paste_mask=None)
        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite Student loss")
        return loss

    @torch.no_grad()
    def equivalence_check(self):
        self.system.eval()
        self.ema_model.eval()
        x, _, mask = self.batch_input(next(iter(self.parser.validloader)))
        reference_stage1 = self.ema_model(x)
        _, reference_alpha = build_alpha_maps(reference_stage1["logits"], mask)
        del reference_stage1
        expected = self.ema_model(x, alpha_map=reference_alpha)
        actual = self.system(x)
        differences = {}
        for name in expected:
            if not torch.isfinite(actual[name]).all():
                raise RuntimeError("Non-finite initial output")
            differences[name] = (expected[name]-actual[name]).abs().max().item()
            torch.testing.assert_close(expected[name], actual[name], atol=0, rtol=0)
        _, soft = self.system.alpha_maps(x)
        assert soft.shape == (3, 1, 512, 512) and not soft.requires_grad and not torch.is_inference(soft)
        assert soft.min() >= .6-1e-6 and soft.max() <= 1.6+1e-6
        torch.testing.assert_close(soft[mask<=0], torch.full_like(soft[mask<=0], 1.2), atol=0, rtol=0)
        torch.testing.assert_close(soft, reference_alpha, atol=0, rtol=0)
        return dict(status="PASS", five_output_max_abs_diff=differences, soft_map_exact=True,
                    teacher_student_input_identical=True, ground_truth_for_alpha=False,
                    alpha_range=[soft.min().item(), soft.max().item()])

    def train_epoch(self, epoch, max_updates=None):
        self.system.train(True)
        self.sampler.epoch = epoch
        self.train_generator.manual_seed(epoch)
        evaluator = d1.iouEval(20, self.device, [0])
        self.optimizer.zero_grad(set_to_none=True)
        total_loss, total_acc, total_iou, samples, updates, accumulated = 0., 0., 0., 0, 0, 0
        started = time.perf_counter()
        teacher_start = self.teacher_identity["state_sha256"]
        first_gradients = None
        for index, batch in enumerate(self.train_loader):
            x, labels, mask = self.batch_input(batch)
            outputs = self.system(x)
            loss = self.full_loss(outputs, labels, mask)
            size = len(x)
            (loss*(size/3.)).backward()
            accumulated += size
            if first_gradients is None:
                first_gradients = [name for name, p in self.model.named_parameters() if p.grad is not None]
            if accumulated == 3:
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.)
                if not torch.isfinite(norm).item():
                    raise RuntimeError("Non-finite Student gradient norm")
                self.optimizer.step()
                self._update_ema()
                self.optimizer.zero_grad(set_to_none=True)
                accumulated = 0
                updates += 1
            with torch.no_grad():
                predictions = d1.original_predictions(outputs["logits"], x[:, :23])
                evaluator.addBatch(predictions, labels)
                total_acc += evaluator.getacc().item()*size
                total_iou += evaluator.getIoU()[0].item()*size
            total_loss += loss.item()*size
            samples += size
            if index % self.ARCH["train"]["report_batch"] == 0:
                self.logger.info(f"D4 train epoch={epoch} microbatch={index+1}/{len(self.train_loader)} updates={updates} lr={self.optimizer.param_groups[0]['lr']:.6e} loss={total_loss/samples:.6f} acc={total_acc/samples:.6f} miou={total_iou/samples:.6f} seconds={time.perf_counter()-started:.1f} paste=0")
            del outputs, loss, predictions, x, labels, mask
            if max_updates is not None and updates >= max_updates:
                break
        if accumulated:
            raise RuntimeError("Incomplete effective batch; do not checkpoint partial optimizer steps")
        self.assert_teacher_fixed()
        assert self.teacher_identity["state_sha256"] == teacher_start
        global_iou, _ = evaluator.getIoU()
        return dict(loss=total_loss/samples, accuracy=total_acc/samples, miou=total_iou/samples,
                    final_global_miou=global_iou.item(), final_global_accuracy=evaluator.getacc().item(),
                    samples=samples, optimizer_updates=updates, seconds=time.perf_counter()-started,
                    parameters_with_gradient_on_first_microbatch=first_gradients)

    def smoke(self):
        torch.cuda.reset_peak_memory_stats(self.device)
        equivalence = self.equivalence_check()
        before = d1.state_digest(self.model.state_dict())
        ema_before = d1.state_digest(self.ema_model.state_dict())
        metrics = self.train_epoch(1, max_updates=3)
        assert d1.state_digest(self.model.state_dict()) != before
        assert d1.state_digest(self.ema_model.state_dict()) != ema_before
        self.assert_teacher_fixed()
        return dict(status="PASS", micro_batch_size=self.ARCH["train"]["micro_batch_size"], effective_batch_size=3,
            equivalence=equivalence, optimizer_updates=metrics["optimizer_updates"], samples=metrics["samples"],
            finite_loss=metrics["loss"], student_changed=True, student_ema_changed=True,
            teacher_unchanged=True, teacher_gradient_none=True, weights_discarded=True,
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
            seconds=metrics["seconds"])

    @torch.no_grad()
    def validate(self, student=None, mode="soft", tag="validation", record_epoch0=False):
        target = self.ema_model if student is None else student
        target.eval()
        self.system.teacher.eval()
        evaluator = d1.iouEval(20, self.device, [0])
        stats = AlphaStatistics()
        samples, total_loss, forward_seconds, seen = 0, 0., 0., 0
        started = time.perf_counter()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        alpha_examples = {}
        for index, batch in enumerate(self.parser.validloader):
            ids = [f"{seq}/{name}" for seq, name in zip(batch[4], batch[5])]
            if ids != self.scan_ids[seen:seen+len(ids)]:
                raise ValueError("Validation scan order changed")
            x, labels, mask = self.batch_input(batch)
            begin.record()
            if mode == "soft":
                hard, soft = self.system.alpha_maps(x)
                outputs = target(x, alpha_map=soft)
            else:
                outputs = target(x)
            end.record()
            end.synchronize()
            forward_seconds += begin.elapsed_time(end)/1000
            if not all(torch.isfinite(value).all().item() for value in outputs.values()):
                raise RuntimeError("Non-finite validation output")
            loss = self.full_loss(outputs, labels, mask)
            predictions = d1.original_predictions(outputs["logits"], x[:, :23])
            evaluator.addBatch(predictions, labels)
            if mode == "soft":
                observed = mask > 0
                values = (hard[observed].cpu().numpy(), soft[observed].cpu().numpy())
                for method, vector in enumerate(values):
                    stats._add(method, vector, np.zeros(len(vector), dtype=np.int64), 0, 1)
                stats.unobserved += int(observed.numel()-observed.count_nonzero().item())
                for sample_index in (0, len(self.scan_ids)//2, len(self.scan_ids)-1):
                    local = sample_index-seen
                    if 0 <= local < len(ids):
                        value = soft[local, 0].cpu().numpy()
                        alpha_examples[str(sample_index)] = value
                        previous = self.output/f"teacher_alpha_epoch0_{sample_index:06d}.npz"
                        if record_epoch0:
                            np.savez_compressed(previous, alpha=value, scan_id=ids[local])
                        elif previous.exists():
                            with np.load(previous, allow_pickle=False) as saved:
                                np.testing.assert_allclose(value, saved["alpha"], atol=1e-6, rtol=1e-5)
                del hard, soft, observed
            size = len(ids)
            total_loss += loss.item()*size
            samples += size
            seen += size
            del outputs, loss, predictions, x, labels, mask
            if (index+1) % 100 == 0 or seen == len(self.scan_ids):
                self.logger.info(f"D4 {tag} batches={index+1}/1357 scans={seen}/4071 partial_miou={evaluator.getIoU()[0].item():.6f} seconds={time.perf_counter()-started:.1f}")
        if seen != 4071:
            raise RuntimeError("Incomplete validation")
        self.assert_teacher_fixed()
        miou, per_class = evaluator.getIoU()
        result = dict(miou=miou.item(), accuracy=evaluator.getacc().item(), loss=total_loss/samples,
            per_class_iou={CLASS_NAMES[cid]:per_class[cid].item() for cid in range(1, 20)},
            confusion_matrix=evaluator.conf_matrix.cpu().tolist(), scans=seen, batches=len(self.parser.validloader),
            seconds=time.perf_counter()-started, forward_cuda_seconds=forward_seconds,
            evaluated_weights="Student EMA" if target is self.ema_model else "Student training weights",
            alpha_mode=mode, teacher_state_sha256=self.teacher_identity["state_sha256"],
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
            alpha_example_sha256={key:hashlib.sha256(value.tobytes()).hexdigest() for key, value in alpha_examples.items()})
        if mode == "soft":
            result["teacher_soft_alpha_statistics"] = {scope:stats.distribution(1, scope=scope) for scope in ("observed", "unobserved", "all")}
            for scope, distribution in result["teacher_soft_alpha_statistics"].items():
                reference = next(row for row in self.reference_run["alpha_statistics"]
                                 if row["mode"] == "D2_Soft" and row["scope"] == scope and row["class_id"] == "")
                if distribution["count"] != reference["count"]:
                    raise RuntimeError("Frozen Teacher alpha cell counts differ from D.2")
                for key in ("mean", "std", "min", "max", "p05", "p50", "p95"):
                    if abs(distribution[key]-reference[key]) > 1e-5:
                        raise RuntimeError(f"Frozen Teacher alpha distribution changed: {scope}/{key}")
            result["teacher_alpha_distribution_matches_D2"] = True
        d1.atomic_json(self.output/f"{tag}.json", result)
        return result

    def checkpoint(self, epoch):
        ema = self.ema_model.state_dict()
        return dict(experiment="Step D.4", epoch=epoch, model=ema, model_train=self.model.state_dict(), model_ema=ema,
            optim=self.optimizer.state_dict(), scheduler=self.scheduler.state_dict(), early_stopping=self.early_stopping.state_dict(),
            best_val_iou=self.best_val_iou, best_val_epoch=self.best_val_epoch, history=self.history,
            teacher_identity=self.teacher_identity, teacher_checkpoint=self.teacher_identity["path"],
            teacher_checkpoint_sha256=self.teacher_identity["sha256"], alpha_table=list(ALPHA_TABLE), teacher_alpha=1.2,
            copy_paste_prob=0., input_features=self.metadata["input_features"], arch_cfg=self.ARCH, data_cfg=self.DATA,
            metadata_id=self.metadata_id, metadata=self.metadata, reproduction_gate_passed=self.gate_passed,
            rng_state=rng_state(), total_train_seconds=self.total_train_seconds)

    def record_history(self):
        fields = ["epoch", "lr", "train_loss", "train_accuracy", "train_miou", "validation_loss", "validation_accuracy", "validation_miou", "best_miou", "best_epoch", "early_stopping_counter", "seconds", "peak_allocated_GiB", "peak_reserved_GiB"]
        d1.write_csv(self.output/"validation_history.csv", fields, [{key:row.get(key) for key in fields} for row in self.history])
        d1.atomic_json(self.output/"training_history.json", self.history)

    def prepare(self):
        if self.gate_passed:
            return
        if d1.state_digest(self.model.state_dict()) != self.teacher_identity["state_sha256"] or d1.state_digest(self.ema_model.state_dict()) != self.teacher_identity["state_sha256"]:
            raise RuntimeError("Student/EMA changed before the epoch0 reproduction gate")
        result = self.validate(tag="epoch0_validation", record_epoch0=True)
        difference = result["miou"]-self.reference["miou"]
        d1.write_csv(self.output/"epoch0_class_comparison.csv", ["Class", "D2_Soft", "D4_Epoch0", "delta"],
            [dict(Class=name, D2_Soft=self.reference["per_class_iou"][name], D4_Epoch0=score,
                  delta=score-self.reference["per_class_iou"][name]) for name, score in result["per_class_iou"].items()])
        if abs(difference) > .0001 or abs(result["miou"]-EXPECTED_D2) > .0001:
            raise RuntimeError(f"Epoch0 D.2b reproduction failed: {result['miou']:.12f}; Student must NOT train")
        self.gate_passed = True
        self.best_val_iou, self.best_val_epoch = result["miou"], 0
        self.early_stopping.update(result["miou"], 0)
        self.history.append(dict(epoch=0, validation_miou=result["miou"], validation_accuracy=result["accuracy"],
            validation_loss=result["loss"], best_miou=self.best_val_iou, best_epoch=0,
            early_stopping_counter=0, seconds=result["seconds"]))
        self.record_history()
        payload = self.checkpoint(0)
        for name in ("epoch0_baseline.path", "best_val.path", "latest.path"):
            atomic_checkpoint(self.output/name, payload)
        self.logger.info(f"Epoch0 reproduction PASS miou={result['miou']:.12f} delta_vs_D2={difference:+.12f}; independent Student initialization checkpoint saved")

    def train(self):
        if not self.gate_passed:
            raise RuntimeError("Formal training requires the epoch0 reproduction gate")
        if self.early_stopping.should_stop(self.next_epoch-1):
            self.logger.info("Resumed checkpoint already satisfies early stopping; continuing to final diagnostics")
            return
        for epoch in range(self.next_epoch, self.ARCH["train"]["max_epochs"]+1):
            started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(self.device)
            rate = self.optimizer.param_groups[0]["lr"]
            train = self.train_epoch(epoch)
            self.scheduler.step()
            valid = self.validate(tag=f"validation_epoch_{epoch:02d}")
            stop = self.early_stopping.update(valid["miou"], epoch)
            improved = valid["miou"] > self.best_val_iou
            if improved:
                self.best_val_iou, self.best_val_epoch = valid["miou"], epoch
            elapsed = time.perf_counter()-started
            self.total_train_seconds += elapsed
            row = dict(epoch=epoch, lr=rate, train_loss=train["loss"], train_accuracy=train["accuracy"], train_miou=train["miou"],
                validation_loss=valid["loss"], validation_accuracy=valid["accuracy"], validation_miou=valid["miou"],
                best_miou=self.best_val_iou, best_epoch=self.best_val_epoch,
                early_stopping_counter=self.early_stopping.bad_epochs, seconds=elapsed,
                peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
                teacher_state_sha256=self.teacher_identity["state_sha256"],
                student_state_sha256=d1.state_digest(self.model.state_dict()),
                student_ema_state_sha256=d1.state_digest(self.ema_model.state_dict()),
                per_class_iou=valid["per_class_iou"], train_final_global_miou=train["final_global_miou"])
            self.history.append(row)
            self.record_history()
            payload = self.checkpoint(epoch)
            atomic_checkpoint(self.output/"latest.path", payload)
            if improved:
                atomic_checkpoint(self.output/"best_val.path", payload)
            self.logger.info(f"D4 epoch={epoch} lr={rate:.6e} train_loss={train['loss']:.6f} train_acc={train['accuracy']:.6f} train_miou={train['miou']:.6f} validation={valid['miou']:.9f} best={self.best_val_iou:.9f}@{self.best_val_epoch} early_stop={self.early_stopping.bad_epochs}/6 seconds={elapsed:.1f} teacher_fixed=True peak_GiB={row['peak_allocated_GiB']:.3f}")
            d1.atomic_json(self.output/"status.json", dict(status="training" if not stop else "early_stopped", epoch=epoch, best_epoch=self.best_val_epoch, best_miou=self.best_val_iou))
            if stop:
                self.logger.info(f"Early stopping after epoch {epoch}; minimum epochs 8, patience 6, significant delta .0003")
                break

    def finalize(self):
        saved = torch.load(self.output/"best_val.path", map_location="cpu", weights_only=False)
        validate_resume(saved, self.ARCH, self.DATA, self.metadata_id, self.teacher_identity)
        self.ema_model.load_state_dict(saved["model_ema"], strict=True)
        self.model.load_state_dict(saved["model_train"], strict=True)
        best = self.validate(tag="best_ema_soft")
        if abs(best["miou"]-saved["best_val_iou"]) > .0001:
            raise RuntimeError("Best checkpoint re-evaluation does not match its saved score")
        raw = self.validate(self.model, tag="best_raw_soft")
        global_alpha = self.validate(mode="global1.2", tag="best_ema_global12")
        epoch0 = json.loads((self.output/"epoch0_validation.json").read_text())
        comparisons = dict(C2_best=self.c2_best_miou, D1b_global12=self.reference_run["reference"]["miou"],
            D2_global12=self.reference_run["results"]["Baseline"]["miou"], D2b_soft=self.reference["miou"],
            D4_epoch0=epoch0["miou"], D4_best=best["miou"], D4_raw=raw["miou"], D4_global12=global_alpha["miou"])
        baseline_classes = self.reference_run["results"]["Baseline"]["per_class_iou"]
        diagnostics = dict(
            improved_vs_D2=best["miou"] > self.reference["miou"],
            recovered_vs_D2_global12={name:best["per_class_iou"][name] >= baseline_classes[name]
                                      for name in ("bicycle", "motorcycle")},
            retained_gains_vs_D2_global12={name:best["per_class_iou"][name] >= baseline_classes[name]
                                           for name in ("truck", "person", "terrain")},
            motorcyclist_above_zero=best["per_class_iou"]["motorcyclist"] > 0,
            student_ema_minus_raw_pp=100*(best["miou"]-raw["miou"]),
            soft_minus_global12_pp=100*(best["miou"]-global_alpha["miou"]),
            teacher_alpha_unchanged=True)
        package = dict(experiment="Step D.4 inference package", teacher=self.system.teacher.state_dict(),
            student=self.ema_model.state_dict(), alpha_table=list(ALPHA_TABLE), teacher_alpha=1.2,
            teacher_identity=self.teacher_identity, arch_cfg=self.ARCH, data_cfg=self.DATA,
            best_epoch=saved["best_val_epoch"], best_miou=best["miou"], metadata=self.metadata)
        atomic_checkpoint(self.output/"inference_package.path", package)
        improved = [name for name in CLASS_NAMES[1:] if best["per_class_iou"][name] > self.reference["per_class_iou"][name]+1e-12]
        worsened = [name for name in CLASS_NAMES[1:] if best["per_class_iou"][name] < self.reference["per_class_iou"][name]-1e-12]
        rows = [dict(Class=name, D2_Soft=self.reference["per_class_iou"][name], D4_Best=best["per_class_iou"][name],
                     delta_pp=100*(best["per_class_iou"][name]-self.reference["per_class_iou"][name])) for name in CLASS_NAMES[1:]]
        d1.write_csv(self.output/"best_class_comparison.csv", list(rows[0]), rows)
        report = ["# D.4 Teacher-conditioned Student Fine-tuning", "", "Exploratory/development evaluation on sequence 08, not an independent test result.",
            "Teacher is the unchanged C.2 epoch-18 best EMA. Only Student is fine-tuned; CP is OFF; no KL/distillation/pseudo-label loss or learned alpha is used.",
            "", f"Best epoch: {saved['best_val_epoch']}; Student EMA mIoU: {best['miou']:.12f}; D.2b: {self.reference['miou']:.12f}; difference: {100*(best['miou']-self.reference['miou']):+.6f} pp.",
            f"Best raw Student mIoU: {raw['miou']:.12f}; best EMA Student with global alpha=1.2: {global_alpha['miou']:.12f}.",
            f"Improved classes ({len(improved)}): {', '.join(improved)}.", f"Worsened classes ({len(worsened)}): {', '.join(worsened)}.",
            "", "| Method | Validation mIoU |", "|---|---|"]
        report += [f"| {name} | {score:.12f} |" for name, score in comparisons.items()]
        report += ["", "D.1b and the D.2 global-1.2 re-evaluation differ slightly numerically; both saved results are shown without modifying either.",
                   "", "| Class | D.2b Soft | D.4 Best EMA | Delta (pp) |", "|---|---|---|---|"]
        report += [f"| {r['Class']} | {r['D2_Soft']:.9f} | {r['D4_Best']:.9f} | {r['delta_pp']:+.6f} |" for r in rows]
        report += ["", f"Training + per-epoch validation wall time: {self.total_train_seconds:.2f} s.",
            f"Early stopping: {self.early_stopping.state_dict()}.",
            f"Peak memory across training: {max((row.get('peak_allocated_GiB', 0) for row in self.history), default=0):.3f} GiB allocated, {max((row.get('peak_reserved_GiB', 0) for row in self.history), default=0):.3f} GiB reserved.",
            f"Best EMA Soft validation: {best['seconds']:.2f} s wall, {best['forward_cuda_seconds']:.2f} s CUDA forward for 4071 scans; raw Soft: {raw['seconds']:.2f} s; EMA global-1.2: {global_alpha['seconds']:.2f} s.",
            f"Diagnostic comparisons against D.2 global-1.2: {json.dumps(diagnostics, sort_keys=True)}.",
            f"Effective batch 3; actual microbatch {self.ARCH['train']['micro_batch_size']}; FP32. Accumulation preserves the nominal sample count, not bitwise-equivalent batch-level Lovasz/Focal normalization. No BatchNorm modules are present.",
            "Teacher alpha examples are checked against epoch0 for every soft-validation pass; distributions and forward times are recorded in the per-validation JSON files.",
            "Validation workers/scan order and original height correction are unchanged. Training workers are recreated per epoch to permit deterministic epoch-boundary resume.",
            "Checkpoint best selection uses strict mIoU maximum, including epoch0 fallback; early stopping alone uses min_delta=.0003, patience=6, min_epochs=8.",
            "The inference_package.path bundles frozen Teacher and best EMA Student with the table/configs; training checkpoints retain optimizer/scheduler/RNG/early-stop state and hashed Teacher identity."]
        (self.output/"comparison.md").write_text("\n".join(report)+"\n")
        result = dict(status="complete", best_epoch=saved["best_val_epoch"], best_miou=best["miou"],
            d2_soft_reference=self.reference["miou"], delta_vs_D2_pp=100*(best["miou"]-self.reference["miou"]),
            improved_classes=improved, worsened_classes=worsened, best_ema_soft=best, best_raw_soft=raw,
            best_ema_global12=global_alpha, early_stopping=self.early_stopping.state_dict(),
            comparisons=comparisons, diagnostics=diagnostics,
            total_training_validation_seconds=self.total_train_seconds, teacher_identity=self.teacher_identity)
        d1b.verify_protected(self.metadata["protected_files"])
        self.assert_teacher_fixed()
        d1.atomic_json(self.output/"results.json", result)
        d1.atomic_json(self.output/"status.json", dict(status="complete", best_epoch=saved["best_val_epoch"], best_miou=best["miou"]))
        return result
