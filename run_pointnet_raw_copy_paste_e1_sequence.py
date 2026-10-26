#!/usr/bin/env python3
"""Durable single-GPU E.1a -> E.1b -> fixed-table evaluation supervisor."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SETUP = ROOT/"logs/SJNet3_PointNet_stepE0c_database_training_setup"
RUNS = (("off","SJNet3_PointNet_stepE1a_raw_matched_cp0"),
        ("on","SJNet3_PointNet_stepE1b_raw_cp_on"))


def status(**values):
    path = SETUP/"sequence_status.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(dict(updated=datetime.now(timezone.utc).isoformat(), supervisor_pid=os.getpid(),**values),indent=2))
    os.replace(temporary,path)
    print(json.dumps(values),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume",action="store_true",help="Resume existing E.1 runs only from identity-matched checkpoints")
    args=parser.parse_args()
    os.chdir(ROOT)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    os.environ.setdefault("OMP_NUM_THREADS","4")
    os.environ.setdefault("MKL_NUM_THREADS","4")
    lock=(SETUP/"e1_sequence.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    from train_pointnet_raw_copy_paste_e1 import check_gate
    check_gate(SETUP/"verification.json")
    # Refuse accidental restarts of either arm before starting any expensive job.
    if not args.resume and any((ROOT/"logs"/name).exists() for _,name in RUNS):
        raise ValueError("An E.1 run exists; use --resume, never overwrite it")
    completed=[]
    for mode,name in RUNS:
        output=ROOT/"logs"/name
        if output.exists():
            result=json.loads((output/"results.json").read_text()) if (output/"results.json").exists() else {}
            if result.get("status")=="complete":
                completed.append(name)
                continue
            if not (output/"latest.path").exists():
                raise ValueError("Existing unfinished run has no resumable checkpoint: "+name)
        config="config/arch/sjnet_pointnet_local3_raw_cp_e1a_cp0.yaml" if mode=="off" else "config/arch/sjnet_pointnet_local3_raw_cp_e1b_cp_on.yaml"
        command=[sys.executable,"-B","-u","train_pointnet_raw_copy_paste_e1.py","--arch_cfg",config,"--log",str(output)]
        if output.exists():
            command += ["--resume",str(output/"latest.path")]
        status(status="training",active=name,completed=completed,command=command)
        code=subprocess.call(command)
        result=json.loads((output/"results.json").read_text()) if (output/"results.json").exists() else {}
        if code or result.get("status")!="complete":
            status(status="stopped",active=name,returncode=code,run_status=result,completed=completed)
            return 1
        completed.append(name)
    for mode,name in RUNS:
        output=ROOT/"logs"/name
        evaluation=output/"post_training_evaluation"
        status(status="evaluating",active=name,completed=completed)
        command=[sys.executable,"-B","-u","verify_raw_copy_paste_e0c.py","--evaluate",str(output/"best_val.path"),"--output",str(evaluation)]
        if not (evaluation/"results.json").exists():
            code=subprocess.call(command)
            if code:
                status(status="evaluation_failed",active=name,returncode=code,completed=completed)
                return code
    comparison=[]
    for mode,name in RUNS:
        output=ROOT/"logs"/name
        train=json.loads((output/"results.json").read_text())
        evaluated=json.loads((output/"post_training_evaluation/results.json").read_text())
        history=json.loads((output/"training_history.json").read_text())
        with (output/"copy_paste_statistics.csv").open() as file:
            cp_rows=list(csv.DictReader(file))
        attempts=sum(int(row["attempted_scans"]) for row in cp_rows)
        successes=sum(int(row["successful_scans"]) for row in cp_rows)
        by_class,boost=Counter(),Counter()
        for row in cp_rows:
            by_class.update(json.loads(row["class_successes"]))
            boost.update(json.loads(row["rare_class_point_delta"]))
        run_summary=dict(cp_attempted_scans=attempts,cp_successful_scans=successes,cp_success_rate=successes/max(1,attempts),
                         cp_class_successes=dict(by_class),rare_class_point_delta=dict(boost),
                         peak_training_allocated_GiB=max(row["peak_allocated_GiB"] for row in history),
                         early_stopping_epoch=train["completed_epoch"] if train["completed_epoch"]<69 else None,
                         first_training_loss=history[0]["training_loss"],last_training_loss=history[-1]["training_loss"],
                         first_validation_loss=history[0]["loss"],last_validation_loss=history[-1]["loss"],
                         final_validation_miou=history[-1]["miou"],best_to_final_validation_delta=history[-1]["miou"]-train["best_miou"],
                         overfitting_note="Inspect full train/validation curves; best-to-final decline is descriptive, not an automatic generalization claim")
        merged=dict(train,evaluation=evaluated,summary=run_summary)
        (output/"results.json").write_text(json.dumps(merged,indent=2))
        comparison.append((name,evaluated))
    off,on=comparison[0][1],comparison[1][1]
    lines=["# E.1 Matched Raw OFF/ON Comparison","", "Best selection: EMA validation global alpha=1 only. Fixed D.2b table is never retuned.","",
           "| Mode | OFF mIoU | ON mIoU | ON - OFF |","|---|---:|---:|---:|"]
    for mode in ("global1","global1.2","D2b_soft"):
        a,b=off["results"][mode]["miou"],on["results"][mode]["miou"]
        lines.append(f"| {mode} | {a:.9f} | {b:.9f} | {b-a:+.9f} |")
    lines += ["", "All 19 per-class IoUs, accuracy, best epoch, early stopping, runtime and peak memory are in results.json.",
              "This is a new matched-control comparison, not an exact repetition of legacy cached C.2. Validation-selected exploratory comparison; not an independent test."]
    deltas={}
    for mode in ("global1","global1.2","D2b_soft"):
        a,b=off["results"][mode],on["results"][mode]
        per_class={name:b["class_iou"][name]-value for name,value in a["class_iou"].items()}
        rare=("bicycle","motorcycle","truck","other-vehicle","person","bicyclist","motorcyclist")
        deltas[mode]=dict(miou=b["miou"]-a["miou"],accuracy=b["accuracy"]-a["accuracy"],class_iou=per_class,
                          rare_class_definition=list(rare),
                          off_rare_miou=sum(a["class_iou"][c] for c in rare)/len(rare),
                          on_rare_miou=sum(b["class_iou"][c] for c in rare)/len(rare))
        lines += ["",f"## {mode} Per-Class Delta","", "| Class | OFF | ON | Delta |","|---|---:|---:|---:|"]
        lines += [f"| {name} | {a['class_iou'][name]:.6f} | {b['class_iou'][name]:.6f} | {value:+.6f} |" for name,value in per_class.items()]
    (SETUP/"comparison.json").write_text(json.dumps(deltas,indent=2))
    for mode,name in RUNS:
        (ROOT/"logs"/name/"comparison.md").write_text("\n".join(lines)+"\n")
    (SETUP/"comparison.md").write_text("\n".join(lines)+"\n")
    status(status="complete",completed=completed)
    return 0


if __name__=="__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        status(status="failed",error=repr(exc))
        raise
