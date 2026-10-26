"""E.0b raw-point CP verification only. No optimizer or formal training."""

import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime
import gzip
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np
import torch
import yaml

from audit_copy_paste_database_e0b import audit_database, write_csv, write_json
from generate_point_features import load_encoder
from lib.utils.laserscan_Polar4 import PRIORITY_MAP, SemLaserScan
from lib.utils.laserscan_Pointnet import build_polar_point_features
from lib.utils.pointnet_copy_paste_e0 import (
    ambiguous_priority_cells, cell_ids_numpy, differences, normalize_handcrafted,
    project_handcrafted, read_bev, seed_all, sha256,
)
from lib.utils.raw_point_copy_paste_e0b import (
    PlacementRejected, PlacementRules, TargetScene, apply_dropout, compatibility_off,
    load_patch, raw_copy_paste, read_train_scan, regenerate_hybrid,
)
from verify_pointnet_copy_paste_e0 import cache_reproduction, dataset_for


def independent_reference(points, packed, learning_map, reference_encoder, device, cpu_encoder=None):
    """Precision-matched MLPs, independent NumPy pooling/projection; CPU diagnostic."""
    hand, mask, _ = project_handcrafted(points, packed, learning_map)
    flat, valid = cell_ids_numpy(points)
    pts, flat = points[valid], flat[valid]
    cells, inverse = np.unique(flat, return_inverse=True)
    raw = packed[valid].astype(np.int64) & 0xFFFF
    order = np.argsort(flat, kind="stable")
    _, starts, counts = np.unique(flat[order], return_index=True, return_counts=True)
    labels = np.zeros(512 * 512, dtype=np.int64)
    lut = np.zeros(max(learning_map) + 1, dtype=np.int64)
    for key, value in learning_map.items():
        lut[key] = value
    for cell, start, count in zip(cells, starts, counts):
        group = raw[order[start:start+count]]
        priority = PRIORITY_MAP[group]
        labels[cell] = lut[group[priority == priority.max()].max()]
    radius = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    theta = np.arctan2(pts[:, 1], pts[:, 0])
    # Match float32 scatter addition order using a separate NumPy accumulation.
    total = np.zeros(len(cells), dtype=np.float32)
    count = np.zeros(len(cells), dtype=np.float32)
    np.add.at(total, inverse, pts[:, 2])
    np.add.at(count, inverse, np.float32(1))
    row, col = flat // 512, flat % 512
    center_r = (row.astype(np.float32) + .5) * np.float32(.1)
    center_t = -np.float32(np.pi) + (col.astype(np.float32) + .5) * np.float32(2*np.pi/512)
    features = np.stack([pts[:, 0]/51.2, pts[:, 1]/51.2,
        (np.clip(pts[:, 2], -5, 15)-1)/5, np.clip(pts[:, 3], 0, 1),
        (radius-center_r)/.1, (theta-center_t)/np.float32(2*np.pi/512),
        np.clip(pts[:, 2]-total[inverse]/count[inverse], -5, 5)/5], axis=1).astype(np.float32)
    with torch.no_grad():
        # NumPy/GPU atan2 differs enough near cancellation to affect small outputs.
        # Keep the authoritative float32 point-input builder; independently test
        # the MLP/pooling/dense reconstruction rather than relaxing tolerances.
        original, original_cells, original_inverse = build_polar_point_features(torch.from_numpy(points).to(device))
        if not np.array_equal(original_cells.cpu().numpy(), cells) or not np.array_equal(original_inverse.cpu().numpy(), inverse):
            raise ValueError("independent_reference_grid_disagreement")
        point_input_max_errors = np.max(np.abs(original.cpu().numpy()-features),axis=0).tolist()
        pointwise = reference_encoder.point_mlp(original).cpu().numpy()
        pooled = np.full((len(cells), 64), -np.inf, dtype=np.float32)
        np.maximum.at(pooled, inverse, pointwise)
        values = reference_encoder.cell_mlp(torch.from_numpy(pooled).to(device)).cpu()
        cpu_dense = None
        if cpu_encoder is not None:
            cpu_pointwise = cpu_encoder.point_mlp(original.cpu()).numpy()
            cpu_pool = np.full((len(cells),64),-np.inf,dtype=np.float32)
            np.maximum.at(cpu_pool,inverse,cpu_pointwise)
            cpu_values=cpu_encoder.cell_mlp(torch.from_numpy(cpu_pool))
            cpu_dense=torch.zeros(16,512*512)
            cpu_dense[:,torch.from_numpy(cells)]=cpu_values.T
            cpu_dense=cpu_dense.reshape(16,512,512)
    dense = torch.zeros(16, 512*512)
    dense[:, torch.from_numpy(cells)] = values.T
    return dict(raw7=hand, point16=dense.reshape(16,512,512), label=torch.from_numpy(labels.reshape(512,512)),
                mask=mask.float(), point_cell_ids=flat, valid_points=valid,
                cpu_cross_device_point16=cpu_dense,
                numpy_vs_authoritative_point_input_max_error_by_channel=point_input_max_errors)


def legacy_projector_comparison(points, packed, result, data):
    state = np.random.get_state()
    scan = SemLaserScan(sem_color_dict=data["color_map"], project=False)
    np.random.set_state(state)
    scan.set_points(points[:, :3], points[:, 3])
    scan.project = True
    start = time.perf_counter()
    scan.set_label(packed.astype(np.int32))
    hand = torch.from_numpy(scan.pseudo_image.transpose(2,0,1))
    mask = torch.from_numpy(scan.proj_mask).bool()
    return dict(raw7=differences(result["raw7"].half(), hand.half(), atol=1e-5,rtol=.002),
                mask=differences(result["mask"][0],mask), seconds=time.perf_counter()-start)


def grid_disagreements(points, features):
    ids, valid = cell_ids_numpy(points)
    if not np.array_equal(valid, features["valid_points"]):
        return {"validity_mismatches": int((valid != features["valid_points"]).sum()), "cell_mismatches": None}
    return {"validity_mismatches": 0, "cell_mismatches": int((ids[valid] != features["point_cell_ids"]).sum())}


def changed_cell_mask(before_points, raw_result, after_features):
    result = after_features["paste_mask"][0].bool().clone()
    ids, valid = cell_ids_numpy(before_points)
    removed = (~raw_result["target_keep"]) & valid
    result.reshape(-1)[torch.from_numpy(np.unique(ids[removed]))] = True
    return result


def save_example(path, before_points, before_labels, patch, raw, before, after, affected):
    np.savez_compressed(path,
        points_before=before_points, packed_labels_before=before_labels, donor_relative=patch,
        donor_transformed=raw["donor_transformed"], donor_visible=raw["donor_visible"],
        points_after=raw["points"], packed_labels_after=raw["labels"], paste_point_mask=raw["paste_points"],
        target_keep=raw["target_keep"], raw7_before=before["raw7"].numpy().astype(np.float16),
        raw7_after=after["raw7"].numpy().astype(np.float16),
        point16_before=before["point16"].numpy().astype(np.float16),
        point16_after=after["point16"].numpy().astype(np.float16), label_before=before["label"].numpy(),
        label_after=after["label"].numpy(), mask_before=before["mask"].numpy(), mask_after=after["mask"].numpy(),
        paste_cell_mask=after["paste_mask"].numpy(), affected_cell_mask=affected.numpy(),
        rotation=np.array(raw["metadata"]["rotation"]), translation=np.array(raw["metadata"]["translation"]))


def visualize(path, points, patch, raw, before, after, affected, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    center = np.array(raw["metadata"]["translation"])
    limit = max(2., float(np.linalg.norm(patch[:,:2],axis=1).max())+1.)
    local_before = points[np.linalg.norm(points[:,:2]-center[:2],axis=1)<limit]
    local_after = raw["points"][np.linalg.norm(raw["points"][:,:2]-center[:2],axis=1)<limit]
    fig = plt.figure(figsize=(17,12))
    for i,(cloud,label) in enumerate([(local_before,"Before raw points (local)"),
                                    (patch,"Donor relative raw patch"),(local_after,"After raw points (local)")]):
        ax=fig.add_subplot(3,4,i+1,projection="3d")
        cloud=cloud[::max(1,len(cloud)//12000)]
        ax.scatter(cloud[:,0],cloud[:,1],cloud[:,2],s=2,c=cloud[:,2],cmap="viridis")
        if i==2:
            new=raw["points"][raw["paste_points"]]
            ax.scatter(new[:,0],new[:,1],new[:,2],c="red",s=4)
        ax.set_title(label,fontsize=10)
        ax.set_xlabel("x (m)");ax.set_ylabel("y (m)");ax.set_zlabel("z (m)")
        ax.zaxis.set_major_locator(MaxNLocator(3))
        ax.tick_params(labelsize=7,pad=1)
        span=np.ptp(cloud[:,:3],axis=0);ax.set_box_aspect(np.maximum(span,.2))
    ax=fig.add_subplot(3,4,4)
    ax.scatter(local_before[:,0],local_before[:,1],s=2,c="gray")
    new=raw["points"][raw["paste_points"]]
    ax.scatter(new[:,0],new[:,1],s=3,c="red");ax.set_aspect("equal");ax.set_title("Paste points highlighted, XY")
    images=[(before["label"],"Before semantic (new deterministic tie)","tab20"),
            (after["label"],"After semantic","tab20"),(after["paste_mask"][0],"Added-point cells","gray"),
            (affected,"Added or occlusion-removed cells","gray"),
            (before["mask"][0],"Before observation","gray"),(after["mask"][0],"After observation","gray"),
            ((after["raw7"]-before["raw7"]).abs().sum(0),"Raw7 absolute change (sum)","magma"),
            ((after["point16"]-before["point16"]).norm(dim=0),"PointNet16 change L2","magma")]
    axes=[]
    for i,(value,label,color)in enumerate(images):
        ax=fig.add_subplot(3,4,i+5);axes.append(ax)
        kw={"vmin":0,"vmax":19}if color=="tab20"else{}
        ax.imshow(value.numpy(),origin="lower",interpolation="nearest",cmap=color,**kw)
        ax.set_title(label,fontsize=9);ax.set_xlabel("azimuth");ax.set_ylabel("radius")
    fig.suptitle(title+"\nRigid insertion + angular-bin occlusion approximation; DB provenance unverified",fontsize=11)
    fig.tight_layout();fig.savefig(path,dpi=120)
    y,x=torch.where(after["paste_mask"][0]>0)
    if len(y):
        wrapped = int(x.max()-x.min()) >= 256
        if wrapped:
            # Display-only re-centering of the seam; never changes CP data.
            for ax,(value,_,_) in zip(axes,images):
                ax.images[0].set_data(torch.roll(value,256,1).numpy())
                ax.set_xlabel("azimuth (+256 display only)")
            y,x=torch.where(torch.roll(after["paste_mask"][0],256,1)>0)
            fig.suptitle(title+"\nPeriodic zoom: display-only azimuth re-centering, raw geometry unchanged",fontsize=11)
        for ax in axes:
            ax.set_xlim(max(0,int(x.min())-15),min(511,int(x.max())+15))
            ax.set_ylim(max(0,int(y.min())-15),min(511,int(y.max())+15))
        suffix="_periodic_zoom.png"if wrapped else"_zoom.png"
        fig.savefig(path.with_name(path.stem+suffix),dpi=140)
    plt.close(fig)


def offline_example(output, dataset, data, raw, features, encoder_hash):
    root=output/"offline_example"
    hand_path=root/"sequences/00/polar_512_prlb/000000.pt"
    point_path=root/"sequences/00/pointnet_16ch/000000.pt"
    hand_path.parent.mkdir(parents=True);point_path.parent.mkdir(parents=True)
    cells=features["cell_ids"]
    entries=[(hand_path,dict(proj_tensor=features["raw7"].half(),mask_t=features["mask"].byte(),labels_t=features["label"].short())),
             (point_path,dict(cell_ids=cells,features=features["point16"].reshape(16,-1)[:,cells].T.half(),
                              feature_dim=16,H=512,W=512,max_radius=51.2,encoder_sha256=encoder_hash,
                              label_tie_policy="maximum raw semantic ID within unchanged priority"))]
    start=time.perf_counter()
    for path,entry in entries:
        with gzip.open(path,"wb")as stream:torch.save(entry,stream)
    write_seconds=time.perf_counter()-start
    ds=dataset_for(root,data,[0],False)
    times=[]
    for _ in range(5):
        start=time.perf_counter();sample=ds[0];times.append(time.perf_counter()-start)
    expected=torch.cat([normalize_handcrafted(features["raw7"].half().float()),features["point16"].half().float()])
    error=differences(sample[0],expected)
    label_error=differences(sample[2],features["label"])
    mask_error=differences(sample[1],features["mask"])
    return dict(write_seconds=write_seconds,read_seconds=times,bytes=sum(p.stat().st_size for p,_ in entries),
                feature_mismatch=error,label_mismatch=label_error,mask_mismatch=mask_error,
                format="actual original C.2 gzip torch caches; one isolated variant, never production data")


def inspect_c2(output):
    run=Path("logs/SJNet3_PointNet_stepC2_local3_gated_lr2e3_bs3")
    ckpt=run/"best_val.path"
    value=torch.load(ckpt,map_location="cpu",weights_only=False)
    scalars={k:v for k,v in value.items()if isinstance(v,(int,float,str,bool))}
    state={k:{"keys":list(v)[:25]}for k,v in value.items()if isinstance(v,dict)}
    info=dict(checkpoint=str(ckpt),checkpoint_sha256=sha256(ckpt),checkpoint_scalars=scalars,
              checkpoint_structures=state,
              missing_historical_files=[p for p in ["train_pointnet_hybrid_gated_local3.py","trainer_PointNet_Gated_StableLR.py"]if not Path(p).exists()],
              present_files=[p for p in ["lib/models/SJNet3_PointNet_Gated_Local3.py","lib/trainer/trainer_PointNet.py",
                  "lib/dataset/SemanticKitti_PointNet.py","lib/dataset/Parser_PointNet.py",
                  "config/arch/sjnet_pointnet_gated_local3_stablelr.yaml","lib/utils/pointnet_d4_training.py"]if Path(p).exists()],
              historical_log="C.2 log starts scratch, lr .002, warmup5/cosine65, batch3, w_decay .005,70 max, ES patience12/min_delta.001/min_epochs30; EMA best saved",
              current_loss="Focal gamma1 ignore0 + aux .30/.15/.10 + Lovasz .50 + mask-weighted boundary .20; w_paste0; current base Trainer aux2/4 decay epochs50..60; historical StableLR implementation absent",
              optimizer="AdamW in current trainer and archived optimizer state, if present",
              scheduler="C.2 log LR preview and lib/utils/pointnet_d4_training.py helper; original StableLR source absent",
              ema="model_ema stored if listed in checkpoint; current trainer default decay .99, exact historical implementation not proven",
              augmentation="cached roll/flip/cell-dropout, PointNet cache; CP hardcoded OFF by Parser",
              executable_exact_recipe_restored=False,formal_training_started=False)
    for key in ("optim","optimizer","optimizer_state_dict"):
        if isinstance(value.get(key),dict):info["optimizer_param_groups"]=value[key].get("param_groups")
    write_json(output/"c2_environment_audit.json",info)
    del value
    return info


def run(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    root=Path.cwd().resolve();output=Path(args.output).resolve()
    if output.parent!=root/"logs" or "stepE0b"not in output.name:
        raise ValueError("Use a dedicated logs/*stepE0b* output")
    if output.exists():
        if not args.reuse_audit or not(output/"database_manifest.json").is_file()or(output/"test_results.json").exists():
            raise ValueError("Existing output refused; --reuse-audit only accepts an audit-only E.0b directory")
    else:output.mkdir()
    if args.audit_dir:
        source=Path(args.audit_dir).resolve()
        for path in source.iterdir():
            if path.is_file()and path.name.startswith(("database_","donor_provenance_","invalid_patches","quality_thresholds")):
                if(output/path.name).exists():raise ValueError("Refusing audit artifact overwrite")
                shutil.copyfile(path,output/path.name)
    os.environ["MPLCONFIGDIR"]=str(output/"matplotlib_cache")
    (output/"runtime_tmp").mkdir(exist_ok=True)
    os.environ["TMPDIR"]=str(output/"runtime_tmp")
    tempfile.tempdir=str(output/"runtime_tmp")
    os.environ["TORCHINDUCTOR_CACHE_DIR"]=str(output/"runtime_tmp/torchinductor")
    def guard(event,arguments):
        if event=="open":
            path,mode,flags=arguments
            writing=(isinstance(mode,str)and any(c in mode for c in "wax+"))or(isinstance(flags,int)and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
            if writing and isinstance(path,(str,bytes,os.PathLike)):
                resolved=Path(os.fsdecode(path)).resolve()
                if not resolved.is_relative_to(output)and str(resolved)!="/dev/null":
                    raise PermissionError(f"E.0b write outside dedicated output denied: {resolved}")
    sys.addaudithook(guard)
    def log(message):
        line=f"{datetime.now().astimezone().isoformat(timespec='seconds')} {message}"
        print(line,flush=True)
        with(output/"execution_log.txt").open("a")as stream:stream.write(line+"\n")
    log("E.0b raw-point verification only. No formal training; unverified DB permitted for audit only.")
    torch.set_num_threads(args.threads);seed_all(args.seed)
    start=time.perf_counter()
    protected_sources=list((root/"lib").rglob("*.py"))+list((root/"config").rglob("*.yaml"))
    protected_sources += [root/"build_database.py",root/"generate_point_features.py",root/"verify_pointnet_copy_paste_e0.py"]
    source_hashes={str(p.relative_to(root)):sha256(p)for p in protected_sources}
    old_logs={str(p):[p.stat().st_size,p.stat().st_mtime_ns]for p in(root/"logs").rglob("*")if p.is_file()and not p.is_relative_to(output)}
    unit=subprocess.run([sys.executable,"-B","-m","unittest","discover","-s","tests","-p","test_raw_point_copy_paste_e0b.py","-v"],capture_output=True,text=True)
    (output/"unit_tests.log").write_text(unit.stdout+unit.stderr)
    if unit.returncode:raise RuntimeError("E.0b tests failed")
    data=yaml.safe_load((root/"config/labels/semantic-kitti.yaml").read_text())
    if(output/"database_manifest.json").exists():
        manifest=json.loads((output/"database_manifest.json").read_text())
        audit=json.loads((output/"database_audit_summary.json").read_text())
    else:manifest,audit=audit_database(args.database,output,data,log)
    dataset=dataset_for(root/"SemanticKitti",data,data["split"]["train"],True)
    encoder,encoder_info=load_encoder(args.encoder,16,torch.device(args.device))
    encoder_hash=sha256(args.encoder);cpu_encoder=copy.deepcopy(encoder).cpu().eval().requires_grad_(False)
    c2=inspect_c2(output)
    paths=[dataset.scan_files[0],dataset.scan_files[len(dataset)//3],dataset.scan_files[2*len(dataset)//3]]
    off=cache_reproduction(dataset,paths,encoder,torch.device(args.device),data,log)
    # Keep the original OFF generator above unchanged; make the new path repeatable.
    torch.use_deterministic_algorithms(True)
    features_report=dict(encoder=args.encoder,encoder_sha256=encoder_hash,encoder_load=encoder_info,
        encoder_eval_frozen=True,compute_dtype="float32, no autocast",cache_save_dtype="float16",
        raw_regeneration_deterministic_algorithms=True,cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
        cache_original_encoder_provenance="not embedded; numerical compatibility tested on three originals",
        independent_reference="Authoritative GPU float32 point-input API + precision-matched point/cell MLP submodules with independent NumPy maximum.at pooling and dense reconstruction, NOT Encoder.forward twice. Independent NumPy7/labels/mask and actual original Polar4; CPU cross-device and NumPy-input precision diagnostics separately retained.",
        off_reproduction=off,successes=[])
    write_json(output/"feature_consistency.json",features_report)
    allowed=defaultdict(list);present=defaultdict(list)
    for row in manifest:
        if row.get("learning_id")in (2,3,4,5,6,7,8):
            present[row["learning_id"]].append(row)
            if row["usable_for_audit"]:allowed[row["learning_id"]].append(row)
    classes=sorted(present);rng=np.random.default_rng(args.seed)
    trials=[];successes=[];labels_report=[];geometry=[];timings=[];represented=set()
    offline=None;first_dropout=None;failure_integrity=[];data_before={}
    (output/"before_after_examples").mkdir(exist_ok=True)
    case_dir=output/"cases";case_dir.mkdir(exist_ok=True)
    rules=PlacementRules()
    while len(trials)<args.min_candidates or len(successes)<args.successes:
        if len(trials)>=args.max_candidates:break
        number=len(trials);cls=classes[number%len(classes)]
        choices=allowed[cls]or present[cls]
        row=choices[int(rng.integers(len(choices)))]
        trial=dict(candidate=number+1,learning_id=cls,donor=row["path"],success=False,reason="pending")
        call_start=time.perf_counter()
        if not row["usable_for_audit"]:
            trial["reason"]="manifest_excluded:"+";".join(row["reasons"])
            trial["seconds"]=time.perf_counter()-call_start;trials.append(trial);continue
        path=dataset.scan_files[int(rng.integers(len(dataset)))]
        seq,frame=dataset.get_seq_frame(path)
        raw_path=Path(path).parent.parent/"velodyne"/(Path(path).stem+".bin")
        label_path=raw_path.parent.parent/"labels"/(raw_path.stem+".label")
        for source_path in (raw_path,label_path,Path(path),Path(dataset.get_pointnet_file(path))):
            if str(source_path)not in data_before:data_before[str(source_path)]=sha256(source_path)
        trial.update(target=str(raw_path),sequence=seq)
        load_start=time.perf_counter()
        points,packed=read_train_scan(raw_path,data["learning_map"],data["split"]["train"])
        scan_seconds=time.perf_counter()-load_start
        scene_start=time.perf_counter();scene=TargetScene(points,packed,data["learning_map"])
        scene_seconds=time.perf_counter()-scene_start
        donor_start=time.perf_counter();patch=load_patch(row["path"],row,data["learning_map"],audit_unverified_provenance=True)
        donor_seconds=time.perf_counter()-donor_start
        try:
            center=scene.sample_ground(cls,rng);yaw=float(rng.uniform(-np.pi,np.pi))
            raw=raw_copy_paste(scene,patch,data["learning_map"],center,yaw,rules)
        except PlacementRejected as error:
            trial["reason"]=str(error);trial["seconds"]=time.perf_counter()-call_start
            trials.append(trial)
            if len(trials)%50==0:log(f"Candidates {len(trials)}; raw feature-validated successes {len(successes)}")
            continue
        after=regenerate_hybrid(raw["points"],raw["labels"],encoder,data["learning_map"],raw["paste_points"])
        before=regenerate_hybrid(points,packed,encoder,data["learning_map"])
        g_before,g_after=grid_disagreements(points,before),grid_disagreements(raw["points"],after)
        if any(g_before.values())or any(g_after.values()):
            trial.update(reason="explicit_cross_backend_boundary_rejection",before_grid=g_before,after_grid=g_after,
                         seconds=time.perf_counter()-call_start)
            trials.append(trial);continue
        reference_start=time.perf_counter()
        ref=independent_reference(raw["points"],raw["labels"],data["learning_map"],encoder,args.device,cpu_encoder)
        reference_seconds=time.perf_counter()-reference_start
        comparison=dict(raw7=differences(after["raw7"].half(),ref["raw7"].half(),atol=1e-5,rtol=.002),
            point16=differences(after["point16"].half(),ref["point16"].half(),atol=1e-5,rtol=.002),
            point16_exact=differences(after["point16"].half(),ref["point16"].half()),
            point16_cpu_cross_device_diagnostic=differences(after["point16"].half(),ref["cpu_cross_device_point16"].half(),atol=1e-5,rtol=.002),
            label=differences(after["label"],ref["label"]),mask=differences(after["mask"],ref["mask"]),
            independent_reference_seconds=reference_seconds,
            numpy_vs_authoritative_point_input_max_error_by_channel=ref["numpy_vs_authoritative_point_input_max_error_by_channel"])
        affected=changed_cell_mask(points,raw,after)
        outside_label=int(((before["label"]!=after["label"])&~affected).sum())
        placed=np.asarray(raw["metadata"]["rotation"])
        translated=np.asarray(raw["metadata"]["translation"])
        expected_xyz=(patch[:,:3].astype(np.float64)@placed.T+translated).astype(np.float32)
        donor_error=float(np.abs(raw["donor_transformed"][:,:3]-expected_xyz).max())
        retained=int(raw["target_keep"].sum())
        target_error=not np.array_equal(raw["points"][:retained],points[raw["target_keep"]])
        target_label_error=not np.array_equal(raw["labels"][:retained],packed[raw["target_keep"]])
        donor_label_error=not np.array_equal(raw["labels"][raw["paste_points"]],patch[raw["donor_visible"],4])
        donor_remission_error=not np.array_equal(raw["points"][raw["paste_points"],3],patch[raw["donor_visible"],3])
        cached7,cached_mask,cached_label=read_bev(path)
        ambiguous=ambiguous_priority_cells(points,packed,data["learning_map"])
        tie_changed=before["label"]!=cached_label
        label_info=dict(candidate=number+1,tie_changed_vs_cache=int(tie_changed.sum()),
            tie_changed_outside_ambiguous=int((tie_changed&~ambiguous).sum()),
            outside_affected_new_tie_label_changes=outside_label,
            point_label_count_match=len(raw["points"])==len(raw["labels"]),
            zero_mask_nonzero_labels=int(((after["mask"][0]==0)&(after["label"]!=0)).sum()),
            stale_pointnet_at_unobserved_values=int(torch.count_nonzero(after["point16"][:,after["mask"][0]==0])),
            paste_at_unobserved_cells=int(((after["paste_mask"][0]>0)&(after["mask"][0]==0)).sum()))
        label_info.update(donor_rigid_transform_max_error=donor_error,
            retained_target_point_error=target_error,retained_target_label_error=target_label_error,
            donor_point_label_error=donor_label_error,donor_remission_error=donor_remission_error)
        independent_paste=np.zeros(512*512,dtype=bool)
        pids,pvalid=cell_ids_numpy(raw["points"])
        independent_paste[pids[raw["paste_points"]&pvalid]]=True
        label_info["paste_cell_mismatch"]=int((after["paste_mask"][0].bool()!=torch.from_numpy(independent_paste.reshape(512,512))).sum())
        comparison.update(candidate=number+1,grid_before=g_before,grid_after=g_after)
        legacy=None
        if len(successes)<args.legacy_reference_cases:
            legacy=legacy_projector_comparison(raw["points"],raw["labels"],after,data)
            comparison["actual_original_projector"]=legacy
        bad=(any(comparison[key]["different_values"]for key in ("raw7","point16","label","mask"))or
             outside_label or label_info["tie_changed_outside_ambiguous"]or label_info["zero_mask_nonzero_labels"]or
             label_info["paste_at_unobserved_cells"]or label_info["paste_cell_mismatch"]or
             donor_error or target_error or target_label_error or donor_label_error or donor_remission_error or
             (legacy is not None and(legacy["raw7"]["different_values"]or legacy["mask"]["different_values"])))
        if bad:
            failure_integrity.append(dict(candidate=number+1,comparison=comparison,label_info=label_info))
            trial.update(reason="integrity_failure_not_hidden",seconds=time.perf_counter()-call_start)
            trials.append(trial);write_json(output/"integrity_failures.json",failure_integrity)
            log(f"Integrity discrepancy at candidate {number+1}; retained for FAIL assessment")
            if len(failure_integrity)>=3:break
            continue
        success_id=len(successes)+1
        metadata=raw["metadata"]
        metadata.update(candidate=number+1,success_id=success_id,target=str(raw_path),donor=row["path"],
            donor_sha256=row["sha256"],donor_provenance="unverified audit only",raw_before_sha256=sha256(raw_path),
            paste_cells=int(after["paste_mask"].sum()),affected_cells=int(affected.sum()))
        case=case_dir/f"case_{success_id:03d}.npz"
        save_start=time.perf_counter();save_example(case,points,packed,patch,raw,before,after,affected)
        write_json(case.with_suffix(".json"),metadata)
        save_seconds=time.perf_counter()-save_start
        if cls not in represented:
            visualize(output/"before_after_examples"/f"class_{cls}_case_{success_id:03d}.png",
                      points,patch,raw,before,after,affected,f"Train {seq}/{frame:06d}; learning class {cls}; donor {Path(row['path']).name}")
            represented.add(cls)
        if offline is None:offline=offline_example(output,dataset,data,raw,after,encoder_hash)
        if first_dropout is None:
            keep=torch.rand(512,512)>.1
            h,l,pm=apply_dropout(after["hybrid24"],after["label"],after["paste_mask"],keep)
            first_dropout=dict(policy="new isolated ignore_dropped",dropped_cells=int((~keep).sum()),
                nonignore_dropped_labels=int((l[~keep]!=0).sum()),nonzero_dropped_features=int(torch.count_nonzero(h[:,~keep])),
                boundary_valid_cells=int(h[23].sum()),baseline_policy="unchanged legacy denoising; still retains labels")
        timing=dict(candidate=number+1,scan_read_seconds=scan_seconds,donor_read_seconds=donor_seconds,
            scene_index_seconds=scene_seconds,placement_seconds=metadata["placement_seconds"],
            collision_visibility_seconds=metadata["collision_visibility_seconds"],**after["timing"],
            archive_write_seconds=save_seconds,reference_seconds=reference_seconds,
            online_seconds=scan_seconds+donor_seconds+scene_seconds+metadata["raw_cp_seconds"]+after["timing"]["total_seconds"])
        timing["measured_candidate_wall_seconds_including_references_archives"]=time.perf_counter()-call_start
        timings.append(timing);geometry.append(metadata);labels_report.append(label_info)
        features_report["successes"].append(comparison)
        trial.update(success=True,reason="success",success_id=success_id,seconds=time.perf_counter()-call_start)
        trials.append(trial);successes.append(metadata)
        log(f"Success {success_id}: class={cls}; points={metadata['added_points']}; cells={metadata['paste_cells']}; independent 7/16/label/mask within tolerance")
        write_json(output/"feature_consistency.json",features_report)
    regressions=[]
    for index in(0,len(dataset)//3,2*len(dataset)//3):
        for seed in(0,17,42):
            seed_all(seed);expected=dataset[index]
            seed_all(seed);actual=compatibility_off(dataset,index)
            errors=[differences(actual[i],expected[i])["different_values"]for i in(0,1,2,14)]
            regressions.append(dict(index=index,seed=seed,mismatch_values=errors))
    class_stats=[]
    for cls in(2,3,4,5,6,7,8,18,19):
        rows=[r for r in trials if r["learning_id"]==cls]
        count=sum(r["success"]for r in rows)
        class_stats.append(dict(learning_id=cls,**{"class":data["labels"][data["learning_map_inv"][cls]]},
            db_filename_class_present=cls in present,candidate_selections=len(rows),successes=count,
            success_rate=count/len(rows)if rows else None,
            reasons=dict(Counter(r["reason"]for r in rows if not r["success"]))))
    write_csv(output/"class_success_statistics.csv",class_stats);write_csv(output/"candidate_statistics.csv",trials)
    write_csv(output/"performance_samples.csv",timings)
    write_json(output/"feature_consistency.json",features_report)
    write_json(output/"label_mask_consistency.json",dict(cases=labels_report,dropout=first_dropout,
        tie_rule="unchanged PRIORITY_MAP, maximum raw semantic ID only on equal priority; OFF delegates original Dataset",
        formal_baseline_loss_changed=False,cp_off_regression=regressions))
    write_json(output/"geometry_validation.json",dict(cases=geometry,approximation=rules.__dict__,
        physical_lidar_simulation=False,rigid_transform=True,no_scaling=True,no_bev_cell_replacement=True))
    source_ok=all(sha256(root/p)==h for p,h in source_hashes.items())
    old_ok=all(Path(p).is_file()and[Path(p).stat().st_size,Path(p).stat().st_mtime_ns]==m for p,m in old_logs.items())
    data_ok=all(sha256(p)==h for p,h in data_before.items())
    db_ok=all(sha256(row["path"])==row["sha256"]for row in manifest)
    off_ok=all(not any(r["mismatch_values"])for r in regressions)
    original_ok=all(not r["pointnet_fp16_with_rounding_tolerance"]["different_values"]and
        not r["handcrafted_fp16_with_rounding_tolerance"]["different_values"]and
        not r["labels"]["different_values"]and not r["mask"]["different_values"]for r in off)
    enough=len(trials)>=args.min_candidates and len(successes)>=args.successes
    failed=bool(failure_integrity)or not enough or not off_ok or not original_ok or not source_ok or not db_ok or not data_ok or not old_ok
    verdict="FAIL"if failed else"CONDITIONAL PASS"
    performance=make_performance(output,timings,trials,offline,len(dataset))
    verification=dict(verdict=verdict,formal_training_started=False,optimizer_created=False,seed=args.seed,
        database_audit=audit,candidates=len(trials),successes=len(successes),targets_met=enough,
        provenance_verified=False,donor_validation_leakage_excluded=False,targets_train_only=True,
        reason="DB source unverified, conservative partial-patch/visibility approximation, new tie/dropout control needed; C.2 historical entry/trainer absent",
        feature_integrity_failures=failure_integrity,cp_off_regression_pass=off_ok,original_cache_reproduction_pass=original_ok,
        source_hashes_unchanged=source_ok,old_logs_size_mtime_unchanged=old_ok,all_database_hashes_unchanged=db_ok,
        read_target_raw_and_cache_hashes_unchanged=data_ok,encoder_hash_unchanged=sha256(args.encoder)==encoder_hash,
        complete_physical_occlusion_proven=False,shared_24ch_cell_mapping=True,
        raw_pointnet_deterministic_algorithms=True,cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
        runtime_seconds=time.perf_counter()-start,cpu_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        GPU=torch.cuda.get_device_name()if args.device.startswith("cuda")else"CPU",performance=performance)
    write_json(output/"verification.json",verification)
    write_json(output/"test_results.json",dict(unit_test_returncode=unit.returncode,unit_test_log="unit_tests.log",
        cp_off=regressions,sample_targets_met=enough,integrity_failure_count=len(failure_integrity)))
    write_json(output/"protected_source_hashes.json",source_hashes)
    write_json(output/"read_data_hashes.json",data_before)
    (output/"raw_copy_paste_report.md").write_text(build_report(verification,features_report,labels_report,class_stats,c2))
    (output/"implementation_notes.md").write_text(implementation_notes())
    log(f"E.0b finished: {verdict}; raw validated {len(successes)}/{len(trials)}. No formal training.")


def make_performance(output,timings,trials,offline,train_scans):
    keys=["scan_read_seconds","donor_read_seconds","scene_index_seconds","placement_seconds",
        "collision_visibility_seconds","preprocessing_seconds","handcrafted7_seconds","pointnet_forward_seconds",
        "label_mask_assembly_seconds","online_seconds"]
    means={k:float(np.mean([r[k]for r in timings]))if timings else None for k in keys}
    peak=max((r["gpu_peak_allocated_bytes"]for r in timings),default=0)
    reserved=max((r["gpu_peak_reserved_bytes"]for r in timings),default=0)
    preprocess_mean=means["online_seconds"]
    result=dict(successful_scan_mean=means,gpu_peak_allocated_bytes=peak,gpu_peak_reserved_bytes=reserved,
        offline=offline,train_scans=train_scans,
        serial_preprocessing_seconds_per_epoch_cp1=preprocess_mean*train_scans if preprocess_mean else None,
        serial_preprocessing_seconds_per_epoch_cp05=preprocess_mean*train_scans*.5 if preprocess_mean else None,
        rejected_candidate_wall_seconds_total=sum(r["seconds"]for r in trials if not r["success"]),
        estimate_only=True,segmentation_training_measured=False)
    write_json(output/"performance_report.json",result)
    (output/"performance_report.md").write_text(f"""# E.0b Performance

No actual training was run. All timing is a microbenchmark on this machine.
Successful online preprocessing means (seconds): `{means}`.
GPU allocated peak: {peak} bytes; reserved peak: {reserved} bytes.
CPU peak RSS is in verification.json; this includes audit/reference/plot memory.
PointNet preprocess/forward are timed separately with CUDA synchronization.
Label/mask assembly is measured separately from handcrafted7 aggregation.

One actual C.2-compatible offline example: `{offline}`.
Offline loading was measured through the unchanged Dataset five times (warm cache).
Raw scan reads also follow safety hashing and therefore use a warm filesystem;
cold-start or network/disk-cache-miss throughput was not measured.
Independent references, before-features, figures, raw archives and DB hashing are
excluded from the online scan timing but retained in candidate wall time.
Rejected candidate time totals are recorded separately; production retry/fallback
policy and class selection would change the accepted-scan cost substantially.

## Epoch Estimate, Not Training Measurement

For {train_scans} train scans, successful preprocessing only, serialized with no
overlap: CP1 about {result['serial_preprocessing_seconds_per_epoch_cp1']} s/epoch;
CP0.5 about {result['serial_preprocessing_seconds_per_epoch_cp05']} s/epoch.
These are illustrative successful-path data-preparation budgets, **not end-to-end
epoch predictions**, and exclude the retry inflation shown above, segmentation
forward/backward/optimizer/EMA/validation, and offline-I/O contention. Actual total
training duration remains unverified. Do not add historical C.2 epoch times from
another GPU to these values as though measured together.

## Online A Versus Offline B

A regenerates from actual changed points, supplies fresh random rigid placements,
needs no augmented bank, but requires CPU I/O/geometry and a frozen GPU encoder.
Use CPU-only DataLoader workers and execute the encoder in the main process;
never initialize CUDA in forked workers. Joint segmentation+encoder memory has NOT
been measured here. Default recommendation is A after provenance and protocol
gates are met; benchmark the full workflow before deciding it fits.

B stores the same raw-derived results in original gzip torch cache schemas under
E.0b only. It avoids training-time encoder/geometry costs but has finite diversity,
disk overhead, an explicit sampling schedule and additional provenance manifests.
Use B if measured full-pipeline throughput/memory makes A unsuitable. The example
size and read/write time are above; it is not a full cache-generation job.
Neither mode resolves unverified donor source or incomplete instance geometry.
""")
    return result


def build_report(v,features,labels,classes,c2):
    table="\n".join(f"| {r['class']} | {r['candidate_selections']} | {r['successes']} | {r['success_rate']} |"for r in classes)
    rows=features["successes"]
    max_hand=max((r["raw7"]["max_absolute_error"]for r in rows),default=0)
    max_point=max((r["point16_exact"]["max_absolute_error"]for r in rows),default=0)
    return f"""# Step E.0b Raw Point Copy-Paste

## Verdict: {v['verdict']}

No formal training, optimizer or model checkpoint was created. All original
Datasets/Trainers/models/configs/data/caches/checkpoints and E.0 results protected.
DB total {v['database_audit']['total_files']}; quality-usable for audit
{v['database_audit']['usable_for_audit']} ({v['database_audit']['usable_fraction']:.2%}).
Full quality statistics, exclusion reasons, measured quantiles and class-specific
rules are in database_audit.md, database_statistics.csv and the complete manifest.

**Donor provenance is unverified**. Code/history names train sequences, but old
files have no source frame/sequence/instance ID. No filename-based inference was
made. Audit-only use was explicitly enabled; the API blocks unknown provenance by
default. We cannot assert absence of validation donors from current DB contents.
Training targets are explicitly restricted to00..07,09,10. No sequence08 target
was loaded or modified; no validation/test data were deliberately selected.

## Actual Raw Algorithm

Load Nx5 relative XYZ/remission/raw-ID patch through a hash-checked quality
manifest. Map raw IDs with the real YAML (11->2,18->4,257->5). Apply a rigid
Z-rotation and translation, no scaling or polar cell displacement. Fit a robust
local plane to observed target ground; place patch minimum-z above that plane by
3cm, reject rough/steep/unsupported/embedded/floating or out-of-radius placements.
This anchors observed patch bottom, NOT a certified complete object's true bottom.

Reject **any** nonground target point inside expanded oriented patch bounds.
Visibility uses2048 azimuth x64 elevation bins, configured sensor elevation limits,
and15cm depth tolerance: nearer target returns hide donor returns, nearer retained
donor returns remove farther target returns; donor self-depth is filtered too.
Require >=80% donor visibility and class-minimum surviving points. Points/labels/
remission stay paired; target packed instance labels are retained, donor instance0
means unknown (not a fabricated original instance). This angular-bin depth buffer
is an approximation, not calibrated HDL64 beam simulation or hidden-surface recovery.

## Feature And Label Integrity

Accepted candidates: {v['successes']} / {v['candidates']} selections, seed{v['seed']}.
All24 channels share the existing PointNet builder's cell mapping. CPU NumPy
legacy-vs-Torch numerical boundary disagreements are explicitly rejected and
counted, never snapped, clipped away or hidden under the old1% occupancy allowance.
The observed angular seam is projected normally; no crop or BEV roll approximates
raw object motion. No cached donor/target features are pasted.

Same changed raw points -> raw7 plus PointNet16 plus mask/labels. Encoder
`{features['encoder']}`, SHA256 `{features['encoder_sha256']}`, frozen eval/no_grad,
float32 compute, float16 cache comparison. Original cache provenance is not embedded;
numerical OFF reproduction tested3 scans. OFF reproduction pass:
{v['original_cache_reproduction_pass']}; seeded original Dataset delegation pass:
{v['cp_off_regression_pass']} (3 scans x3 seeds, all23 features/labels/masks/paste).

Independent reference reuses the authoritative GPU float32 point-input API and
precision-matched MLP submodules, with separate NumPy maximum.at pooling and dense
reconstruction instead of calling Encoder.forward again; it independently
recomputes NumPy7/labels/mask and
compares the actual original Polar4 projector on selected cases. Not the same
end-to-end function called twice. Maximum FP16 hand7 error {max_hand}; PointNet16
error {max_point}. Exact counts and atol1e-5/rtol0.002 counts both saved; no enlarged
tolerance selected to hide alignment problems. Unobserved PointNet is zero because
there are no raw points, not missing-feature imputation. Empty normalized z keeps
the baseline value-0.2. Max-z inverse remains input[:,0]*5+1.

CPU cross-device diagnostics retain
{sum(r['point16_cpu_cross_device_diagnostic']['different_values']for r in rows)}
values outside the same tolerance. These are NOT dropped from the JSON. Both CPU
matmul/BN rounding and NumPy-vs-GPU atan2 input differences were isolated in prior
diagnostic runs. The primary reference matches the actual generator's device and
precision without relaxing tolerances; independent pooling/index reconstruction
and the original handcrafted projector still validate the raw-data correspondence.
New CUDA regeneration requires deterministic algorithms and
CUBLAS_WORKSPACE_CONFIG=:4096:8, including deterministic cell z-sums. Original
cached OFF reproduction runs before enabling this new execution policy.

Tie break: original raw-label PRIORITY_MAP unchanged, largest raw semantic ID only
within equal highest priority. This rule is deterministic and input-order invariant.
It changes {sum(r['tie_changed_vs_cache']for r in labels)} baseline cache labels across
successful cases; changes outside ambiguous-priority cells:
{sum(r['tie_changed_outside_ambiguous']for r in labels)}. It is **not silently applied
to CP OFF**, which delegates the original Dataset. A fair E.1 comparison needs an
additional matched OFF control using the new raw/tie/dropout pipeline.

## Class Successes

| Class | Candidate selections | Validated successes | Rate |
| --- | --- | --- | --- |
{table}

Class7/8 standalone riders are excluded because original vehicle alignment is
absent. Pole/sign are absent from the DB filename classes and were not invented.
Failures include quality gates, ground, collision, visibility and numerical-grid
boundary checks; all are retained in candidate_statistics.csv. Quality-excluded
donor selections and geometric attempts are distinguished by reason, not counted
as successful placements. Hard integrity discrepancies: {len(v['feature_integrity_failures'])}.

## Dropout And Loss

Baseline untouched: main/aux Focal and Lovasz ignore label0, boundary BCE alone
is observation-mask-weighted. Legacy dropout retains semantic labels (denoising).
New isolated helper defaults to ignore_dropped: zero7/16/mask/paste AND set dropped
labels0, without changing the loss. An explicit legacy_denoising option preserves
intentional hidden-cell supervision. Tests verify ignored main/aux gradients and
mask validity. No production Dataset or loss setting was changed. E.1 must choose
and record this policy, apply it to a matched OFF control, derive boundary targets
from post-geometry PRE-dropout labels before masking dropped cells, and avoid
calling a policy change a CP-only ablation. Clearing labels first can introduce
artificial Sobel boundaries at neighboring kept cells; the test proves dropped-cell
BCE masking, not complete neighbor-target semantics.

## Artifacts, Performance And E.1 Gates

Every validated success has cases/case_NNN.npz +JSON with before/after raw points,
packed labels, relative/transformed donor, transform, visibility/point/cell masks,
before/after7+16, observation and labels. Representative classes have3D/raw/BEV PNGs
and zooms. Removed-background affected cells are distinct from added-point paste cells.
Offline example uses the unchanged original Dataset/cache schema under E.0b only.

Performance: performance_report.md/JSON/CSV (all requested stages, GPU allocated/
reserved, CPU RSS, successful-path illustrative epoch overhead and retry caveats).
Recommend online A once gates are met; B only if full-workflow memory/throughput
measurement warrants a cache bank. No real C.2 training duration was measured.

Missing historical C.2 files: {c2['missing_historical_files']}. Model, cache Dataset,
Parser, saved config and best checkpoint are present, but exact historical Trainer
source/executable recipe is not restored. Inspect c2_environment_audit.json.
Before E.1: prove/rebuild source-traceable train-only donors in a NEW DB, resolve
partial-instance and rider alignment limitations, validate physical placement/
occlusion assumptions, approve tie/dropout protocol plus matched OFF control,
restore historical C.2 training recipe, benchmark joint model+encoder workflow.

Safety: source hashes {v['source_hashes_unchanged']}; original log size/mtime
{v['old_logs_size_mtime_unchanged']}; all DB hashes unchanged
{v['all_database_hashes_unchanged']}; read raw/label/cache hashes unchanged
{v['read_target_raw_and_cache_hashes_unchanged']}. No training was started.
"""


def implementation_notes():
    return """# Implementation Notes

New isolated sources: audit_copy_paste_database_e0b.py,
lib/utils/raw_point_copy_paste_e0b.py, verify_raw_copy_paste_e0b.py,
tests/test_raw_point_copy_paste_e0b.py, memo/stepE0b_raw_point_copy_paste.md.
No production Dataset/Trainer/Model/config import was added or modified.
Existing E.0 helpers are reused read-only for cached/OFF comparisons and normalization.
Explicit manifest provenance/quality/hash gates precede diagnostic donor use.
Only raw rigid insertion is used, not legacy feature CP or E.0 polar cell replacement.
Source-less DB diagnostics must never become formal-training authorization.

Run the audit independently or run verification in a fresh E.0b logs directory.
Use --reuse-audit only for an audit-only output, not an existing result.
--audit-dir can copy a previous immutable audit to a new output for a rerun.
All generated files go under the dedicated output; Python write guard denies
writes elsewhere. No old DB/cache/checkpoint is rebuilt. Tests use synthetic fixtures
only to cover boundary/rejection behavior; they are not included in real success counts.

Diagnostic runs retained under separate new E.0b directories:
- raw_copy_paste_probe: all-NumPy point-input precision mismatch, not training.
- raw_copy_paste_probe2: short successful CPU/pooling comparison probe.
- raw_copy_paste_cpu_reference_diagnostic: CPU/GPU near-zero numerical differences.
- raw_copy_paste_atomic_sum_diagnostic: same-GPU unordered z-sum differences.
Final raw generation uses deterministic CUDA z-sums and precision-matched independent
NumPy max pooling; CPU/NumPy-input diagnostic values remain in final JSON.
No tolerance or quality gate was loosened to hide these differences.
The final execution log also retains a resolved temporary-cache write denial;
all runtime caches are now confined to runtime_tmp/ inside the E.0b output.
Authoritative final measurements and verdict are verification.json, not a prior probe.
"""


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="logs/SJNet3_PointNet_stepE0b_raw_copy_paste_integrity")
    parser.add_argument("--database",default="copy_paste_database")
    parser.add_argument("--reuse-audit",action="store_true")
    parser.add_argument("--audit-dir")
    parser.add_argument("--encoder",default="point_encoder_runs/best_point_encoder.pth")
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--threads",type=int,default=2)
    parser.add_argument("--seed",type=int,default=20261023)
    parser.add_argument("--min-candidates",type=int,default=120)
    parser.add_argument("--successes",type=int,default=30)
    parser.add_argument("--max-candidates",type=int,default=1400)
    parser.add_argument("--legacy-reference-cases",type=int,default=3)
    args=parser.parse_args()
    try:run(args)
    except Exception:
        output=Path(args.output)
        if output.is_dir()and output.parent.resolve()==Path.cwd()/"logs"and"stepE0b"in output.name and not(output/"verification.json").exists():
            with(output/"execution_log.txt").open("a")as stream:stream.write(traceback.format_exc())
        raise
