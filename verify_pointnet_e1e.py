"""E.1e train-only paired preflight. Never write to an E.1a-d experiment."""
import gc
import json
import subprocess
import sys
import time
from collections import Counter
from itertools import groupby
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from train_pointnet_paper_e1e import (Trainer, PREFLIGHT, config, pins, sha256, atomic_json, torch,
                                       deterministic_encoder, base_config, CLASSES)
from build_donors_e1e import DB, LABELS, SETTINGS, audit_donor
from compare_pointnet_paper_e1d import NAMES, locate
from lib.utils.raw_point_copy_paste_e1e import augment, extra_tally, summary
from lib.utils import raw_point_copy_paste_paper_e1d as old
from lib.utils.raw_point_copy_paste_e0c import TRAIN_SEQUENCES
from trainer_PointNet_RawCopyPaste_E1 import state_digest
from trainer_PointNet_SharedSoftAlpha_D5 import nested_digest
import verify_pointnet_paper_e1d as old_verify


def inventory():
    names=("best_val.path","latest.path","training_history.json","results.json","cp_stats.csv","train_log.txt")
    return {str(p):sha256(p) for exp in NAMES for name in names for p in [locate(exp)/name]}


def audit(trainer):
    provenance=json.loads((DB/"provenance.json").read_text())
    assert provenance["code_sha256"]==sha256("build_donors_e1e.py") and provenance["labels_sha256"]==sha256(LABELS)
    assert tuple(provenance["sequences"])==TRAIN_SEQUENCES and provenance["source_scans"]==19130
    lookup=np.full(65536,-1)
    for k,v in trainer.data["learning_map"].items(): lookup[k]=v
    rows=json.loads((DB/"manifest.json").read_text())
    key=lambda r:(r["source_sequence"],r["source_frame"])
    for (seq,frame),group in groupby(sorted(rows,key=key),key):
        seqdir=Path("SemanticKitti/sequences")/seq
        p=seqdir/"velodyne"/(frame+".bin");l=seqdir/"labels"/(frame+".label")
        hashes=dict(bin_sha256=sha256(p),packed_sha256=sha256(l))
        points=np.fromfile(p,np.float32).reshape(-1,4);packed=np.fromfile(l,np.uint32)
        for row in group:
            assert row["source"]==hashes and row["code_sha256"]==provenance["code_sha256"]
            audit_donor(DB/row["file"],row,points,packed,lookup)
    for cid in CLASSES[5:]:
        for quality in ("accept","reject"):
            assert len(list((DB/"examples").glob(f"{cid}_{quality}_*.png")))>=2
    legacy=trainer.dataset.database
    for row in legacy.rows:
        assert sha256(legacy.root/row["file"])==row["sha256"]
    return dict(new_verified=len(rows),legacy_verified=len(legacy.rows),groups={c:len(trainer.donors.groups[c]) for c in CLASSES})


def baseline():
    folder=locate(NAMES[-1])
    rows=json.loads((folder/"paper_cp_epoch_history.json").read_text())
    counts=Counter()
    for row in rows.values():
        counts.update({k:v for k,v in row.items() if k not in ("epoch","next_batch") and not k.startswith("mean_")})
    rec=json.loads((folder/"results.json").read_text())
    return dict(epochs=len(rows),best_epoch=rec["best_epoch"],best_miou=rec["best_miou"],**summary(counts))


def measure(trainer,indices,budgets,include_old=False):
    cp=trainer.arch["copy_paste"];mapping=trainer.data["learning_map"]
    keys=(["E1d"] if include_old else [])+list(budgets)
    counts={k:Counter() for k in keys};times=Counter();attempt_times=Counter();rows=[]
    examples=[]
    old_cp=base_config()[0]["copy_paste"]
    for i,idx in enumerate(indices):
        raw=trainer.dataset[int(idx)]
        row={"scan":raw["scan_id"],"index":int(idx)}
        # Alternate method order. Disk reads and feature regeneration are outside CP timing for both arms.
        for k in (keys if i%2==0 else list(reversed(keys))):
            s=dict(raw,stats=dict(raw["stats"]))
            torch.cuda.synchronize();started=time.perf_counter()
            with deterministic_encoder():
                if k=="E1d":
                    old.augment(s,trainer.dataset.database,mapping,old_cp,trainer.device)
                else:
                    augment(s,trainer.donors,mapping,dict(cp,search_budget=k),trainer.device)
            torch.cuda.synchronize();dt=time.perf_counter()-started
            st=s["stats"];times[k]+=dt;attempt_times[k]+=dt*st["attempted_scans"]
            old.tally(counts[k],st);extra_tally(counts[k],st)
            row[str(k)]=dict(attempt=st["attempted_scans"],n=sum(st["class_successes"].values()),
                             trials=st["candidate_trials"],seconds=dt)
            if len(keys)==2 and include_old and k!="E1d" and row[str(k)]["n"]==3:
                examples.append((int(idx),[p["learning_id"] for p in s["placement"]],
                                 sum(p["blocked_cells"] for p in s["placement"])))
        assert len({row[str(k)]["attempt"] for k in keys})==1
        rows.append(row)
    return ({str(k):dict(summary(counts[k]),seconds_per_scan=times[k]/len(indices),
                 seconds_per_attempted_scan=attempt_times[k]/max(1,counts[k]["attempted_scans"])) for k in keys},rows,examples)


def main():
    PREFLIGHT.mkdir(exist_ok=True,parents=True)
    if (PREFLIGHT/"verification.json").exists():
        raise FileExistsError("Completed preflight exists")
    suites=["test_pointnet_e1e.py","test_pointnet_paper_e1d.py","test_basic_raw_copy_paste_e1c.py",
            "test_pointnet_training_e1.py","test_raw_point_copy_paste_e0b.py","test_raw_point_copy_paste_e0c.py"]
    for suite in suites:
        with (PREFLIGHT/(suite+".log")).open("w") as log:
            subprocess.run([sys.executable,"-B","-m","unittest","discover","-s","tests","-p",suite,"-v"],
                           stdout=log,stderr=subprocess.STDOUT,check=True)
    prior=inventory()
    arch,data=config();initial_pins=pins(arch)
    atomic_json(PREFLIGHT/"historical_e1d.json",baseline())
    t=Trainer(arch,data,PREFLIGHT)
    db=audit(t)
    print("Source/hash audit PASS",db,flush=True)
    # Disjoint, stratified random train subsets. Never select budgets on sequence 08.
    rng=np.random.default_rng(0);pilot=[];held=[]
    for seq in TRAIN_SEQUENCES:
        idx=[i for i,p in enumerate(t.dataset.files) if p.parent.parent.name==seq]
        draw=rng.choice(idx,28,replace=False)
        pilot.extend(draw[:8]);held.extend(draw[8:])
    assert not set(pilot)&set(held)
    # Measure the reference before tuning; warm both device projection and filesystem.
    reference,_,_=measure(t,pilot,[],True)
    limits=arch["copy_paste"]["calibration_budgets"]
    calibration,_,_=measure(t,pilot,limits)
    passing=[b for b in limits if calibration[str(b)]["rate_3"]>=.70]
    budget=min(passing) if passing else max(limits,key=lambda b:calibration[str(b)]["rate_3"])
    atomic_json(PREFLIGHT/"calibration.json",dict(seed=0,indices=list(map(int,pilot)),baseline=reference,
                calibration=calibration,selected=budget,rule="smallest >=70%, otherwise highest measured rate"))
    print("Calibration",[(k,r["rate_3"]) for k,r in calibration.items()],"selected",budget,flush=True)
    arch["copy_paste"]["search_budget"]=budget
    metrics,rows,examples=measure(t,held,[budget],True)
    old_r,r=metrics["E1d"],metrics[str(budget)]
    atomic_json(PREFLIGHT/"paired_comparison.json",dict(e1d=old_r,e1e=r,seed=0,rows=rows,split="train"))
    assert r["rate_3"]>old_r["rate_3"] and r["attempted_scans"]==old_r["attempted_scans"]
    assert all(r[f"class_{c}"]>0 for c in CLASSES)
    print("Paired trials",r["attempted_scans"],"three-object rates",old_r["rate_3"],r["rate_3"],flush=True)
    # Independently replay a real triple and verify all channels/labels and protected cells
    # using the E.1d diagnostic with its output directory redirected to E.1e.
    idx,_,_=max(examples,key=lambda e:(e[2]>0,len(set(e[1])&set(CLASSES[5:])),e[2]))
    sample=t.dataset[idx];raw,packed=sample["points"],sample["packed"];t.prepare(sample)
    replay=t.dataset[idx];t.prepare(replay)
    for name in ("points","packed","paste_points"): np.testing.assert_array_equal(sample[name],replay[name])
    assert sample["placement"]==replay["placement"] and sample["stats"]["class_rejections"]==replay["stats"]["class_rejections"]
    facade=SimpleNamespace(arch=arch,data=data,device=t.device,encoder=t.encoder,dataset=SimpleNamespace(database=t.donors))
    with patch.object(old_verify,"PREFLIGHT",PREFLIGHT):
        old_verify.verify_example(facade,sample,raw,packed)
    # Exercise the actual bounded loop for every stopping count, including attempted failures.
    seen=set()
    for ix in held:
        for b in (1,2,3,budget):
            s=t.dataset[int(ix)]
            with deterministic_encoder(): augment(s,t.donors,data["learning_map"],dict(arch["copy_paste"],search_budget=b),t.device)
            if s["stats"]["attempted_scans"]: seen.add(len(s["placement"]))
        if seen=={0,1,2,3}: break
    assert seen=={0,1,2,3}
    assert not t.validation.is_train and t.validation.copy_paste_prob==0
    picked=[i for i,_,_ in examples[:3]]
    batch=[t.dataset[i] for i in picked]
    loss,_,_,diag=t.update(batch)
    model,ema,optim=(state_digest(t.model.state_dict()),state_digest(t.ema_model.state_dict()),nested_digest(t.optimizer.state_dict()))
    assert np.isfinite(loss) and diag["gradient_norm"]>0 and diag["input_shape"]==[3,24,512,512]
    assert all(int(s["step"])==1 for s in t.optimizer.state.values()) and model!=ema and t.initial_hash not in (ema,model)
    assert not t.encoder.training and t.encoder_state_hash==state_digest(t.encoder.state_dict())
    assert all(not p.requires_grad and p.grad is None for p in t.encoder.parameters())
    count=t.counts.copy();scheduler=t.scheduler.state_dict();scaler=t.scaler.state_dict()
    t.save(0,1,dict(meters=dict(cp=[s["stats"] for s in batch])))
    initial=t.initial_hash;del t;gc.collect();torch.cuda.empty_cache()
    resumed=Trainer(arch,data,PREFLIGHT,PREFLIGHT/"latest.path")
    assert resumed.next_batch==1 and resumed.next_epoch==0 and resumed.counts==count
    assert state_digest(resumed.model.state_dict())==model and state_digest(resumed.ema_model.state_dict())==ema
    assert nested_digest(resumed.optimizer.state_dict())==optim and scaler==resumed.scaler.state_dict() and scheduler==resumed.scheduler.state_dict()
    resumed_loss,*_=resumed.update(next(iter(resumed.loader(0,1))))
    assert np.isfinite(resumed_loss)
    assert prior==inventory() and initial_pins==pins(arch)
    atomic_json(PREFLIGHT/"verification.json",dict(status="PASS",inputs=initial_pins,budget=budget,source_audit=db,
                protected_artifacts=prior,calibration="calibration.json",paired="paired_comparison.json",unit_suites=suites,
                scan=sample["scan_id"],loss=loss,diagnostics=diag,counts_tested=sorted(seen),resume_loss=resumed_loss,
                initial_sha256=initial,frozen_encoder=resumed.encoder_state_hash))
    print("E.1e PASS",loss,resumed_loss,flush=True)


if __name__=="__main__":
    main()
