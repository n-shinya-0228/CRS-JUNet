#!/usr/bin/env python3
"""E.0c mandatory preparation gates. Never starts formal training on a failure."""

import argparse
import ast
from collections import Counter
import copy
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import torch
import yaml
from loguru import logger

sys.path.insert(0,str(Path(__file__).resolve().parent/"lib/trainer"))
from trainer_PointNet_RawCopyPaste_E1 import (Trainer, atomic_json, augmentation, deterministic_encoder,
                                              state_digest, validation_dataset)
from audit_copy_paste_database_e0c import audit
from generate_point_features import load_encoder
import evaluate_pointnet_fixed_alpha_sensitivity as d1
from lib.dataset.Parser_PointNet import bev_collate_fn
from lib.models.SJNet3_PointNet_Gated_Local3_ClassAlpha import (
    CLASS_NAMES, SJNet3_PointNet_Gated_Local3_ClassAlpha, build_alpha_maps)
from lib.utils.laserscan_Pointnet import build_polar_point_features
from lib.utils.laserscan_Polar4 import PRIORITY_MAP
from lib.utils.pointnet_copy_paste_e0 import differences, read_bev, sha256
from lib.utils.raw_point_copy_paste_e0b import regenerate_hybrid
from lib.utils.raw_point_copy_paste_e0c import ENCODER_SHA256, CLASSES
from verify_pointnet_copy_paste_e0 import cache_reproduction, dataset_for

OUTPUT = Path("logs/SJNet3_PointNet_stepE0c_database_training_setup")
C2 = Path("logs/SJNet3_PointNet_stepC2_local3_gated_lr2e3_bs3/best_val.path")


def canonical_reference(points, packed, mapping, encoder, device):
    """Independent per-cell projection and NumPy max pooling on the authoritative GPU grid."""
    with torch.no_grad():
        tensor = torch.from_numpy(points).to(device)
        inputs, cells, inverse = build_polar_point_features(tensor)
        valid = ((tensor[:,:2].square().sum(1).sqrt()>0) & (tensor[:,:2].square().sum(1).sqrt()<51.2)).cpu().numpy()
        ids, inv = cells.cpu().numpy(), inverse.cpu().numpy()
        pointwise = encoder.point_mlp(inputs).cpu().numpy()
        pooled = np.full((len(ids),64),-np.inf,np.float32)
        np.maximum.at(pooled,inv,pointwise)
        values = encoder.cell_mlp(torch.from_numpy(pooled).to(device)).cpu().numpy()
    pts = points[valid]
    raw = packed[valid]&65535
    order = np.argsort(inv,kind="stable")
    _, starts, counts = np.unique(inv[order],return_index=True,return_counts=True)
    dense = np.zeros((7,512*512),np.float32)
    labels = np.zeros(512*512,np.int64)
    for cid, start, count in zip(ids,starts,counts):
        group = pts[order[start:start+count]]
        semantic = raw[order[start:start+count]]
        hi,lo = group.max(0),group.min(0)
        dense[:,cid] = [hi[2], group[:,2].mean(),hi[3],count/100.,hi[2]-lo[2],hi[0]-lo[0],hi[1]-lo[1]]
        priority = PRIORITY_MAP[semantic]
        labels[cid] = mapping[int(semantic[priority==priority.max()].max())]
    point = np.zeros((16,512*512),np.float32)
    point[:,ids] = values.T
    mask = np.zeros((1,512*512),np.float32)
    mask[:,ids] = 1
    return dict(raw7=torch.from_numpy(dense.reshape(7,512,512)),point16=torch.from_numpy(point.reshape(16,512,512)),
                label=torch.from_numpy(labels.reshape(512,512)),mask=torch.from_numpy(mask.reshape(1,512,512)))


def recipe_audit(arch,data):
    historical = OUTPUT/"historical_trainer_PointNet_Gated_StableLR.py"
    git_source=subprocess.check_output(["git","show","c0a0716c:lib/trainer/trainer_PointNet_Gated_StableLR.py"])
    original = ast.parse(git_source)
    if ast.dump(original)!=ast.dump(ast.parse(historical.read_text())):
        raise ValueError("Saved reference differs from the actual historical Git source")
    extracted = ast.parse(Path("lib/utils/pointnet_c2_recipe_e1.py").read_text())
    old_class = next(n for n in original.body if isinstance(n,ast.ClassDef) and n.name=="Trainer")
    new_class = next(n for n in extracted.body if isinstance(n,ast.ClassDef) and n.name=="C2Recipe")
    names = {"_mix_losses","_compute_boundary_gt","_focal_loss_map","_build_ema_model","_update_ema"}
    matched = {}
    for name in names:
        old = next(n for n in old_class.body if isinstance(n,ast.FunctionDef) and n.name==name)
        new = next(n for n in new_class.body if isinstance(n,ast.FunctionDef) and n.name==name)
        matched[name] = ast.dump(old)==ast.dump(new)
    for name in ("FocalLoss","SobelFilter","EarlyStopping","build_scheduler"):
        old = next(n for n in original.body if getattr(n,"name",None)==name)
        new = next(n for n in extracted.body if getattr(n,"name",None)==name)
        matched[name] = ast.dump(old)==ast.dump(new)
    if not all(matched.values()):
        raise ValueError("Historical recipe AST differs")
    checkpoint = torch.load(C2,map_location="cpu",weights_only=False)
    saved = checkpoint["arch_cfg"]["train"]
    fields = ("lr","min_lr","max_epochs","wup_epochs","cos_epochs","w_decay","batch_size","epsilon_w","loss","early_stopping")
    for key in fields:
        if saved[key] != arch["train"][key]:
            raise ValueError("C.2 config mismatch: "+key)
    if checkpoint["data_cfg"] != data or sha256(C2) != "23b581814e920afe8f463666f1cbf6d7a3e1b3d131577bef82239bb7fdb5fc9b":
        raise ValueError("C.2 checkpoint identity mismatch")
    evidence = dict(status="PASS",recipe="Reconstructed C.2 Recipe; recovered identical core methods, not complete training entry restoration",
                    historical_commit="c0a0716c8e0b270b5ed98162bfe80874635bec0a",
                    historical_trainer_sha256=sha256(historical),ast_equal=matched,
                    historical_git_blob_sha256=__import__("hashlib").sha256(git_source).hexdigest(),
                    reference_snapshot_note="One extra trailing newline only; complete AST equals Git",
                    optimizer=checkpoint["optim"]["param_groups"][0].copy(), verified_config={k:saved[k] for k in fields},
                    ema_decay=.99,ema_source="historical Trainer default, no override in saved config",
                    loss_weights=dict(main=1.,aux2=.3,aux4=.15,aux8=.1,lovasz=.5,boundary=.2,paste=0.),
                    auxiliary_schedule="aux2/aux4 only, linear epoch 50..60 then zero; aux8 stays .10",
                    amp="BF16 autocast + CUDA GradScaler; validation FP32", gradient_clip=5.,
                    normalization="model unchanged; segmentation train() including existing BatchNorm, encoder always eval()",
                    decay_scope="all segmentation parameters, including norm/bias/unused parameters, as historical AdamW",
                    cudnn="deterministic=True, benchmark=True as actual historical CUDA branch",
                    dropout="model .3; feature roll/flip; probability .5 of .1 cell dropout; semantic labels retained",
                    policy_differences=["raw re-generation/shared GPU grid", "deterministic priority/raw-ID tie-break",
                                        "explicit separate per-scan augmentation/CP/order random streams", "epoch/mid-epoch RNG resume"],
                    validation="unchanged cached C.2 eval split, ignore0, FP32 EMA, height -0.8, KNN not called")
    evidence["optimizer"].pop("params")
    atomic_json(OUTPUT/"c2_recipe_audit.json",evidence)
    text = "# Reconstructed C.2 Recipe\n\n"+json.dumps(evidence,indent=2)+"\n"
    (OUTPUT/"c2_recipe_audit.md").write_text(text)
    (OUTPUT/"reconstruction_report.md").write_text(text)
    return evidence


def raw_integrity(arch,data):
    from lib.dataset.SemanticKitti_PointNet_RawCopyPaste_E1 import SemanticKitti
    encoder,_ = load_encoder(arch["dataset"]["pointnet"]["encoder"],16,torch.device("cuda"))
    reference = copy.deepcopy(encoder).eval().requires_grad_(False)
    legacy = dataset_for(Path("SemanticKitti"),data,data["split"]["train"],False)
    # Preserve the old non-deterministic cache-reproduction procedure as a separate regression.
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    old_regression = cache_reproduction(legacy,[legacy.scan_files[0],legacy.scan_files[len(legacy)//3],
                                              legacy.scan_files[2*len(legacy)//3]],encoder,torch.device("cuda"),data,logger.info)
    torch.set_float32_matmul_precision(precision)
    if any(row[key]["different_values"] for row in old_regression for key in
           ("mask", "labels", "pointnet_fp16_with_rounding_tolerance", "handcrafted_fp16_with_rounding_tolerance")):
        raise ValueError("Original C.2 cache regression failed")
    cp = dict(arch["copy_paste"],probability=1.)
    dataset = SemanticKitti("SemanticKitti",data,cp,True)
    successes, counts, attempts, rejects, regressions = [],Counter(),Counter(),Counter(),[]
    examples = OUTPUT/"raw_examples"
    examples.mkdir(exist_ok=True)
    for index in range(3000):
        dataset.epoch = index//len(dataset)
        sample = dataset[(index*173)%len(dataset)]
        if index<3:
            before = sample["before_points"] if sample["before_points"] is not None else sample["points"]
            packed_path = dataset.files[sample["index"]].parent.parent/"labels"/(dataset.files[sample["index"]].stem+".label")
            original_packed = np.fromfile(packed_path,dtype=np.uint32)
            with deterministic_encoder():
                regenerated = regenerate_hybrid(before,original_packed,encoder,data["learning_map"])
                independent = canonical_reference(before,original_packed,data["learning_map"],reference,torch.device("cuda"))
            comparisons = compare(regenerated,independent)
            cache = dataset.files[sample["index"]].parent.parent/"polar_512_prlb"/(dataset.files[sample["index"]].stem+".pt")
            old7,oldmask,oldlabel = read_bev(cache)
            regressions.append(dict(scan=sample["scan_id"],independent=comparisons,
                cache_differences=dict(raw7=differences(regenerated["raw7"].half().float(),old7,atol=1e-5,rtol=.002),
                                      label=differences(regenerated["label"],oldlabel),mask=differences(regenerated["mask"],oldmask))))
        attempts.update(sample["stats"]["class_attempts"])
        rejects.update(sample["stats"]["rejection_reasons"])
        if not sample["stats"]["successful_scans"]:
            continue
        with deterministic_encoder():
            result = regenerate_hybrid(sample["points"],sample["packed"],encoder,data["learning_map"],sample["paste_points"])
            independent = canonical_reference(sample["points"],sample["packed"],data["learning_map"],reference,torch.device("cuda"))
        comparisons = compare(result,independent)
        cls = next(iter(sample["stats"]["class_successes"]))
        counts[cls] += 1
        entry = dict(scan_id=sample["scan_id"],placement=sample["placement"],stats=sample["stats"],differences=comparisons)
        successes.append(entry)
        if counts[cls] == 1 or len(successes)<=5:
            np.savez_compressed(examples/f"case_{len(successes):03d}.npz",points=sample["points"],packed=sample["packed"],
                                 before_points=sample["before_points"],paste_points=sample["paste_points"],
                                 raw7=result["raw7"].numpy(),point16=result["point16"].numpy(),labels=result["label"].numpy(),
                                 mask=result["mask"].numpy(),paste_mask=result["paste_mask"].numpy())
        if len(successes)%10==0:
            logger.info(f"New donor raw integrity successes={len(successes)} | classes={dict(counts)}")
        if len(successes)>=50 and all(counts[str(cid)] for cid in CLASSES):
            break
    passed = len(successes)>=50 and all(counts[str(cid)] for cid in CLASSES)
    result = dict(status="PASS" if passed else "FAIL",successes=successes,class_counts=dict(counts),
                  class_attempts=dict(attempts),legacy_cache_regression=old_regression,
                  off_regression=regressions,candidate_attempts=sum(attempts.values()),scan_attempts=index+1,
                  successful_scans=len(successes),scan_success_rate=len(successes)/(index+1),
                  candidate_success_rate=len(successes)/max(1,sum(attempts.values())),rejection_reasons=dict(rejects),
                  reference="Independent per-cell NumPy 7ch/label/mask + NumPy max pooling, authoritative float32 PointNet GPU inputs/grid",
                  comparison_to_old_cache="Differences expected from label ties and NumPy/GPU boundary rounding; common new pipeline in BOTH arms",
                  visibility="2048x64 angular-bin approximation, not exact physical LiDAR")
    atomic_json(OUTPUT/"raw_copy_paste_integrity.json",result)
    with (OUTPUT/"class_success_statistics.csv").open("w") as file:
        file.write("learning_id,attempts,successes\n")
        for cid in CLASSES:
            file.write(f"{cid},{attempts[str(cid)]},{counts[str(cid)]}\n")
    if not passed:
        raise ValueError("Missing required real CP successes")
    del encoder,reference,result
    gc.collect()
    torch.cuda.empty_cache()


def compare(actual,expected):
    result = {}
    for name in ("raw7","point16","label","mask"):
        a,b = actual[name],expected[name]
        if name in ("raw7","point16"):
            a,b = a.half().float(),b.half().float()
        result[name] = differences(a,b,atol=1e-5 if name=="raw7" else 0,rtol=.002 if name=="raw7" else 0)
        if result[name]["different_values"]:
            raise ValueError("Raw integrity mismatch: "+name+" "+json.dumps(result[name]))
    return result


def smoke(arch,data):
    records,hashes = {},[]
    for enabled in (False,True):
        output = OUTPUT/("smoke_on" if enabled else "smoke_off")
        output.mkdir(exist_ok=True)
        trainer = Trainer(arch,data,"SemanticKitti",output,logger,enabled)
        hashes.append(trainer.initial_hash)
        steps, success, scan_order, ordinary_seeds = [],0,[],[]
        started = time.perf_counter()
        trainer.set_epoch_weights(0)
        loader = iter(trainer.loader(0))
        # Continue past three updates if the ON sample has not actually pasted yet.
        for index in range(12):
            update_started = time.perf_counter()
            batch = next(loader)
            scan_order.extend(row["scan_id"] for row in batch)
            ordinary_seeds.extend(row["augmentation_seed"] for row in batch)
            loss,_,_,diagnostics = trainer.update(batch)
            success += sum(row["stats"]["successful_scans"] for row in batch)
            steps.append(dict(loss=loss,**diagnostics,cp_successes=sum(row["stats"]["successful_scans"] for row in batch),
                              wall_seconds=time.perf_counter()-update_started,
                              peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30))
            logger.info(f"Smoke CP={enabled} update={index+1} | loss={loss:.6f} | pasted scans={success}")
            if index>=2 and (not enabled or success):
                break
        del loader
        if len(steps)<3 or (enabled and not success):
            raise ValueError("Smoke did not exercise actual CP")
        if not any(p.grad is not None for p in trainer.model.parameters()):
            raise ValueError("Segmentation model has no gradients")
        model_hash,ema_hash = state_digest(trainer.model.state_dict()),state_digest(trainer.ema_model.state_dict())
        scale, scheduler = trainer.scaler.state_dict(),trainer.scheduler.state_dict()
        from trainer_PointNet_SharedSoftAlpha_D5 import nested_digest
        optimizer_digest=nested_digest(trainer.optimizer.state_dict())
        if any(int(state["step"])!=len(steps) for state in trainer.optimizer.state.values()):
            raise ValueError("Smoke did not perform actual optimizer updates")
        trainer.save(0,len(steps),partial=dict(smoke=True))
        elapsed = time.perf_counter()-started
        encoder_hash = trainer.encoder_state_hash
        if state_digest(trainer.encoder.state_dict()) != encoder_hash:
            raise ValueError("Encoder mutated")
        initial_hash = trainer.initial_hash
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        resumed = Trainer(arch,data,"SemanticKitti",output,logger,enabled,output/"latest.path")
        if (state_digest(resumed.model.state_dict()) != model_hash or state_digest(resumed.ema_model.state_dict()) != ema_hash or
                resumed.scaler.state_dict()!=scale or resumed.scheduler.state_dict()!=scheduler or resumed.next_batch!=len(steps) or
                nested_digest(resumed.optimizer.state_dict())!=optimizer_digest):
            raise ValueError("Smoke checkpoint/resume mismatch")
        batch = next(iter(resumed.loader(0,resumed.next_batch)))
        resumed_loss,_,_,diagnostics = resumed.update(batch)
        records["on" if enabled else "off"] = dict(status="PASS",updates=steps,resumed_update_loss=resumed_loss,
              checkpoint_roundtrip=True,encoder_frozen=True,initial_state_sha256=initial_hash,
              native_batch_size=3,no_accumulation=True,seconds=elapsed,seconds_per_update=elapsed/len(steps),
              steady_seconds_per_update=float(np.median([row["wall_seconds"] for row in steps[1:]])),
              actual_cp_successes=success)
        records["on" if enabled else "off"].update(first_nine_scan_order=scan_order[:9],first_nine_ordinary_seeds=ordinary_seeds[:9])
        atomic_json(OUTPUT/"gpu_smoke_test.json",records)
        del resumed,batch
        gc.collect()
        torch.cuda.empty_cache()
    if len(set(hashes)) != 1:
        raise ValueError("Scratch initial states differ between arms")
    if (records["off"]["first_nine_scan_order"]!=records["on"]["first_nine_scan_order"] or
            records["off"]["first_nine_ordinary_seeds"]!=records["on"]["first_nine_ordinary_seeds"]):
        raise ValueError("Ordinary augmentation or scan order differs between OFF/ON")
    return hashes[0],records


def evaluation_signature(arch,data):
    paths = list(Path("lib/models").glob("*.py"))+[Path("lib/dataset/SemanticKitti_PointNet.py"),
            Path("lib/dataset/Parser_PointNet.py"),Path("evaluate_pointnet_fixed_alpha_sensitivity.py")]
    def function_ast(path,name):
        tree=ast.parse(Path(path).read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        return ast.dump(node)
    validation=validation_dataset("SemanticKitti",data,arch)
    manifest=d1.dataset_manifest(validation,"SemanticKitti")[2]
    return dict(files={str(p):sha256(p) for p in paths},data=data,
                sensor=arch["dataset"]["sensor"],checkpoint_sha256=sha256(C2),validation_manifest_sha256=manifest,
                evaluation_ast=function_ast(__file__,"evaluation"),
                validation_dataset_ast=function_ast("lib/trainer/trainer_PointNet_RawCopyPaste_E1.py","validation_dataset"),
                torch=str(torch.__version__),float32_matmul_precision="high",cudnn_deterministic=True,cudnn_benchmark=True)


def write_setup_report(gate):
    database=json.loads((OUTPUT/"database_quality_audit.json").read_text())
    raw=json.loads((OUTPUT/"raw_copy_paste_integrity.json").read_text())
    smoke_results=json.loads((OUTPUT/"gpu_smoke_test.json").read_text())
    validation=json.loads((OUTPUT/"validation_regression.json").read_text())
    statistics=database["statistics"]
    candidates=sum(row["candidates"] for row in statistics.values())
    passed=sum(row["passed"] for row in statistics.values())
    errors={key:max(row["differences"][key]["max_absolute_error"] for row in raw["successes"])
            for key in ("raw7","point16","label","mask")}
    lines=["# E.0c Setup Report", "", f"Final verdict: **{gate['verdict']}**. See verification.json for mandatory conditions.","",
           "## Database","", f"{passed} accepted / {candidates} primary-class candidates ({passed/candidates:.2%}).",
           "Every accepted donor was verified against ALL original observed instance points and packed labels; Train only, validation donors zero.","",
           "| Class | Accepted | Candidates |", "|---|---:|---:|"]
    lines += [f"| {CLASS_NAMES[int(cid)]} | {row['passed']} | {row['candidates']} |" for cid,row in statistics.items()]
    lines += ["", "Nonzero instance IDs scoped to sequence/frame; mixed/zero instances excluded whole. Riders excluded: no authoritative paired-vehicle metadata.",
              "Shape/support checks cannot prove hidden physical surface completeness. Thresholds fixed before validation; no validation-based donor tuning.","",
              "## Raw Integrity","",f"{len(raw['successes'])} successful pasted scans / {raw['scan_attempts']} forced-CP scans; "
              f"{raw['candidate_attempts']} candidates. Scan success {raw['scan_success_rate']:.2%}; candidate success {raw['candidate_success_rate']:.2%}.",
              f"Class successes: {raw['class_counts']}",f"Primary maximum absolute errors after FP16 quantization: {errors}.",
              "7ch comparisons use atol=1e-5/rtol=.002 (FP16 rounding); 16ch/label/mask require exact equality.",
              "Independent per-cell projection/labels/masks and NumPy max pooling, not Encoder.forward twice.",
              "Legacy OFF cache regression: three originals passed. New OFF independent regeneration: three passed; legacy tie/rounding differences retained in raw_copy_paste_integrity.json.",
              "Rigid yaw+translation; ground fit; non-ground OBB rejection; approximate 2048x64 angular depth occlusion. No scaling/shearing or cached feature copying.","",
              "## Training Compatibility","",
              "Reconstructed C.2 Recipe with Git-recovered core methods (AST-equal to c0a0716c). Not a complete original training-entry/random-stream reproduction.",
              "AdamW .002/.005; warmup5 + cosine65/min1e-5; max70; native3; EMA .99; BF16/GradScaler; clip5. Aux2/4 decay epoch50..60 only.",
              "Class-weighted focal gamma1 + Lovasz .5 + aux .30/.15/.10 + boundary .20; paste weight0. Existing model normalization/mask/skips/all heads unchanged.",
              "Both arms: same raw path/tie rule/order/ordinary RNG, legacy denoising (labels kept under dropped inputs; boundary loss post-dropout mask). CP RNG separate.",
              "Fixed encoder: SHA256 98d3ebe4c120b1cc44b3e830cece8d2f4fd3f2401a40a1fb558e811bea161e97; eval/no_grad, main CUDA process only.",
              "Encoder uses highest FP32 matmul and deterministic algorithms in a scoped context; segmentation retains historical high matmul precision. CUBLAS_WORKSPACE_CONFIG=:4096:8.",
              f"Same random initialization hash in both arms: {gate['initial_state_sha256']}.",
              f"Smoke OFF/ON: {[(mode,len(row['updates']),row['checkpoint_roundtrip']) for mode,row in smoke_results.items()]} (plus resumed updates); actual ON pastes confirmed.",
              "Raw/EMA/optimizer/scheduler/scaler resume equality verified, encoder unchanged/no grads, segmentation finite gradients. Smoke weights discarded before formal seed0 reset.","",
              "## Full Validation Regression","", "All 4071 sequence08 scans; unchanged original cached validation, FP32 EMA, height -0.8 correction, ignore0, KNN OFF."]
    lines += [f"- {mode}: mIoU={row['miou']:.12f}, accuracy={row['accuracy']:.12f}" for mode,row in validation["results"].items()]
    lines += [f"C.2 delta={validation['delta_c2']:+.12f}; D.2b delta={validation['delta_soft']:+.12f}; tolerance=1e-4.","",
              "## E.1 Execution","", "Only after this PASS: E.1a OFF scratch -> E.1b ON scratch -> best EMA alpha1/1.2/fixed-D2b evaluations, sequential single GPU.",
              "ON: probability .5, max1 object, max10 candidates, balanced eligible-class trials. OFF effective probability0. Configs differ ONLY in enabled.",
              "Best selection ONLY normal alpha1 EMA validation mIoU. ES patience12/min_delta.001/min_epoch30. No mid-run retuning.",
              "Training/evaluation results remain pending until the respective jobs finish. See sequence_status.json and each train_log.txt, not this setup report, for live status.",
              "Small-smoke timing estimates (cold startup separately reported; not guaranteed full-run times):", "", json.dumps(gate["performance_estimates"],indent=2),"",
              "## Commands","", "```bash", ".venv/bin/python -B -u build_copy_paste_database_e0c.py",
              ".venv/bin/python -B -u verify_raw_copy_paste_e0c.py",
              ".venv/bin/python -B -u run_pointnet_raw_copy_paste_e1_sequence.py",
              "# Resume without overwriting existing logs:",".venv/bin/python -B -u run_pointnet_raw_copy_paste_e1_sequence.py --resume", "```", "",
              "The resolved preparation precision mismatch and repeated tests after logging/provenance edits remain in execution_log.txt for traceability.",
              "No independent test or generalization claim: same validation sequence has been used repeatedly; existing D.2b table is never retuned."]
    (OUTPUT/"setup_report.md").write_text("\n".join(lines)+"\n")


@torch.no_grad()
def evaluation(arch,data,checkpoint,output):
    from torch.utils.data import DataLoader
    from lib.utils.ioueval import iouEval
    output.mkdir(parents=True,exist_ok=True)
    saved = torch.load(checkpoint,map_location="cpu",weights_only=False)
    state = saved["model_ema"]
    model = SJNet3_PointNet_Gated_Local3_ClassAlpha(in_channels=24,nclasses=20,drop=.3)
    model.load_state_dict(state,strict=True)
    model.to("cuda").eval().requires_grad_(False)
    if sum(p.numel() for p in model.parameters()) != 28876586:
        raise ValueError("C.2 parameter count mismatch")
    dataset = validation_dataset("SemanticKitti",data,arch)
    loader = DataLoader(dataset,batch_size=3,num_workers=4,shuffle=False,collate_fn=bev_collate_fn,pin_memory=True)
    evaluators = {mode:iouEval(20,torch.device("cuda"),[0]) for mode in ("global1","global1.2","D2b_soft")}
    wall,scans = time.perf_counter(),0
    forward_seconds = Counter()
    import hashlib
    files,scan_ids,manifest_hash = d1.dataset_manifest(dataset,"SemanticKitti")
    expected_meta = json.loads(Path("logs/SJNet3_PointNet_stepD2_class_fixed_alpha/metadata.json").read_text()) if Path("logs/SJNet3_PointNet_stepD2_class_fixed_alpha/metadata.json").exists() else json.loads(Path("logs/SJNet3_PointNet_stepD2_class_fixed_alpha/results.json").read_text())["metadata"]
    if manifest_hash != expected_meta["validation_manifest_sha256"]:
        raise ValueError("Validation manifest differs from the frozen D.2 comparison")
    for index,batch in enumerate(loader):
        x = torch.cat((batch[0],batch[1].float()),dim=1).to("cuda")
        labels = batch[2].long().to("cuda")
        model.set_alpha(1.2)
        begin = time.perf_counter()
        stage1 = model(x)
        if index==0:
            assert set(stage1)=={"logits","aux2","aux4","aux8","boundary"}
            assert all(v.shape==(3,1 if k=="boundary" else 20,512,512) for k,v in stage1.items())
        _,soft = build_alpha_maps(stage1["logits"],x[:,23:24])
        torch.cuda.synchronize()
        forward_seconds["global1.2"] += time.perf_counter()-begin
        evaluators["global1.2"].addBatch(d1.original_predictions(stage1["logits"],x[:,:23]),labels)
        del stage1
        for mode in ("global1","D2b_soft"):
            model.set_alpha(1. if mode=="global1" else 1.2)
            begin = time.perf_counter()
            outs = model(x,alpha_map=soft if mode=="D2b_soft" else None)
            torch.cuda.synchronize()
            forward_seconds[mode] += time.perf_counter()-begin
            evaluators[mode].addBatch(d1.original_predictions(outs["logits"],x[:,:23]),labels)
            del outs
        scans += len(x)
        if index%100==0:
            logger.info(f"3-mode validation {index}/{len(loader)} | scans={scans}/4071 | elapsed={time.perf_counter()-wall:.1f}s")
    if scans!=4071 or d1.state_digest(model.state_dict())!=d1.state_digest(state):
        raise ValueError("Incomplete evaluation or model mutated")
    results = {}
    for mode,evaluator in evaluators.items():
        miou, per_class = evaluator.getIoU()
        results[mode] = dict(miou=float(miou),accuracy=float(evaluator.getacc()),scans=scans,
                            class_iou={CLASS_NAMES[cid]:float(per_class[cid]) for cid in range(1,20)},
                            forward_seconds=forward_seconds[mode]+(forward_seconds["global1.2"] if mode=="D2b_soft" else 0))
    result = dict(status="PASS",checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),weights="EMA",
                  best_epoch=saved["best_val_epoch"],results=results,wall_seconds=time.perf_counter()-wall,
                  peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,validation_manifest_sha256=manifest_hash,
                  soft_table="Fixed pre-existing D.2b table; no retuning, no checkpoint reselection",
                  caveat="Exploratory validation-selected results, not independent test generalization")
    atomic_json(output/"results.json",result)
    del saved,state,model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate",help="Evaluate one E.1 best EMA after training")
    parser.add_argument("--output")
    parser.add_argument("--reuse-validation",action="store_true",help="Reuse a complete full-set evaluation ONLY with identical evaluation inputs/code signature")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    logger.add(OUTPUT/"execution_log.txt",mode="a")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=True
    arch = yaml.safe_load(Path("config/arch/sjnet_pointnet_local3_raw_cp_e1.yaml").read_text())
    data = yaml.safe_load(Path("config/labels/semantic-kitti.yaml").read_text())
    if args.evaluate:
        evaluation(arch,data,Path(args.evaluate),Path(args.output))
        return
    gate = dict(verdict="FAIL",formal_training_started=False,mandatory_conditions={})
    atomic_json(OUTPUT/"verification.json",gate)
    try:
        protected = {str(p):sha256(p) for folder in ("lib","config") for p in Path(folder).rglob("*") if p.suffix in (".py",".yaml")}
        protected.update({str(p):sha256(p) for p in (C2,Path(arch["dataset"]["pointnet"]["encoder"]),
                          Path("copy_paste_database_e0c_trainonly/manifest.json"),Path("copy_paste_database_e0c_trainonly/provenance.json"),
                          Path("train_pointnet_raw_copy_paste_e1.py"),Path("run_pointnet_raw_copy_paste_e1_sequence.py"),
                          Path("verify_raw_copy_paste_e0c.py"),Path("audit_copy_paste_database_e0c.py"),
                          Path("build_copy_paste_database_e0c.py"),Path("tests/test_pointnet_training_e1.py"),
                          Path("tests/test_raw_point_copy_paste_e0c.py"))})
        unit = subprocess.run([sys.executable,"-B","-m","unittest","discover","-s","tests","-p","test_*e1.py","-v"],capture_output=True,text=True)
        (OUTPUT/"e1_unit_tests.log").write_text(unit.stdout+unit.stderr)
        if unit.returncode or "Ran 0 tests" in unit.stderr:
            raise ValueError("E.1 CPU tests failed")
        unit2=subprocess.run([sys.executable,"-B","-m","unittest","discover","-s","tests","-p","test_raw_point_copy_paste_e0c.py","-v"],capture_output=True,text=True)
        (OUTPUT/"e0c_unit_tests.log").write_text(unit2.stdout+unit2.stderr)
        if unit2.returncode or "Ran 0 tests" in unit2.stderr:
            raise ValueError("E.0c CPU tests failed")
        gate["mandatory_conditions"]["cpu_tests"] = True
        audit(arch["copy_paste"]["donor_database"],data,OUTPUT)
        gate["mandatory_conditions"]["all_donors_train_instance_source_verified"] = True
        recipe_audit(arch,data)
        (OUTPUT/"c2_training_recipe_audit.md").write_bytes((OUTPUT/"c2_recipe_audit.md").read_bytes())
        (OUTPUT/"c2_reconstruction_report.md").write_bytes((OUTPUT/"reconstruction_report.md").read_bytes())
        gate["mandatory_conditions"]["c2_major_recipe_verified"] = True
        raw_integrity(arch,data)
        gate["mandatory_conditions"]["raw_7_16_label_mask_integrity_and_off_regression"] = True
        initial_hash,smokes = smoke(arch,data)
        gate["mandatory_conditions"]["gpu_off_on_three_updates_resume_identical_init"] = True
        signature = json.loads(json.dumps(evaluation_signature(arch,data)))
        signature_path = OUTPUT/"c2_validation_execution_signature.json"
        if args.reuse_validation:
            if not signature_path.exists() or json.loads(signature_path.read_text()) != signature:
                raise ValueError("Cannot reuse validation: evaluation inputs/code changed")
            baseline=json.loads((OUTPUT/"c2_validation_regression/results.json").read_text())
            if baseline["checkpoint_sha256"]!=sha256(C2) or any(r["scans"]!=4071 for r in baseline["results"].values()):
                raise ValueError("Cannot reuse incomplete/wrong-checkpoint validation")
            logger.info("Reusing full 4071-scan validation: code, model, cache manifest and label signature unchanged")
        else:
            atomic_json(signature_path,signature)
            baseline = evaluation(arch,data,C2,OUTPUT/"c2_validation_regression")
        delta_c2 = baseline["results"]["global1"]["miou"]-.508252934833837
        delta_soft = baseline["results"]["D2b_soft"]["miou"]-.5097046791309
        passed = abs(delta_c2)<=1e-4 and abs(delta_soft)<=1e-4
        atomic_json(OUTPUT/"validation_regression.json",dict(baseline,delta_c2=delta_c2,delta_soft=delta_soft,status="PASS" if passed else "FAIL"))
        if not passed:
            raise ValueError("C.2/D.2 full validation regression mismatch")
        if any(sha256(path)!=value for path,value in protected.items()):
            raise ValueError("Protected inputs changed during verification")
        gate["mandatory_conditions"]["full_validation_regression_and_inputs_unchanged"] = True
        gate.update(verdict="PASS",protected_inputs=protected,initial_state_sha256=initial_hash,
                    validation_manifest_sha256=baseline["validation_manifest_sha256"],
                    performance_estimates={mode:dict(seconds_per_update=v["steady_seconds_per_update"],
                        startup_inclusive_smoke_seconds_per_update=v["seconds_per_update"],
                        approximate_epoch_training_hours=v["steady_seconds_per_update"]*6376/3600,
                        approximate_70epoch_training_hours=v["steady_seconds_per_update"]*6376*70/3600,
                        excludes_validation_and_early_stopping=True) for mode,v in smokes.items()})
    except BaseException as exc:
        gate["error"] = repr(exc)
        logger.exception("E.0c failed; formal E.1 training MUST NOT start")
        atomic_json(OUTPUT/"verification.json",gate)
        raise
    atomic_json(OUTPUT/"verification.json",gate)
    write_setup_report(gate)
    logger.info("E.0c PASS: all mandatory conditions satisfied; formal E.1a may start")


if __name__ == "__main__":
    main()
