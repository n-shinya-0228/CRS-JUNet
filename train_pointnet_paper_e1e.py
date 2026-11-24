"""Scratch E.1e with an immutable, train-only calibration and verification gate."""
import argparse
import csv
import json
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from train_pointnet_paper_raw_copy_paste_e1d import (
    Trainer as E1d, FILES as E1D_FILES, CONFIG as BASE, LABELS, read_config as base_config,
    sha256, atomic_json, deterministic_encoder, logger, yaml, torch,
)
from lib.utils.raw_point_copy_paste_e1e import CombinedDonors, CLASSES, augment, extra_tally, summary
from tensorboardX import SummaryWriter

CONFIG = Path("config/arch/sjnet_pointnet_e1e_paper3_9class.yaml")
RUN = Path("logs/SJNet3_PointNet_stepE1e_raw_cp_paper3_9class")
PREFLIGHT = RUN.with_name(RUN.name+"_preflight")
FILES = [*E1D_FILES, str(CONFIG), "build_donors_e1e.py", "lib/utils/raw_point_copy_paste_e1e.py",
         "verify_pointnet_e1e.py", "tests/test_pointnet_e1e.py", "train_pointnet_paper_e1e.py",
         "lib/utils/raw_point_copy_paste_basic_e1c.py", "lib/utils/laserscan_Pointnet.py",
         "generate_point_features.py","verify_raw_copy_paste_e0c.py"]


def config(budget=None):
    arch, data = base_config()
    overlay = yaml.safe_load(CONFIG.read_text())
    if Path(overlay["base"]) != BASE:
        raise ValueError("expected E.1d base")
    cp = dict(overlay["copy_paste"])
    policy = arch["copy_paste"]["paper"]
    policy["allowed_background"].update(cp.pop("additional_background"))
    policy["max_overlap"].update(cp.pop("additional_overlap"))
    arch["copy_paste"].update(cp, search_budget=budget or cp["calibration_budgets"][0])
    return arch, data


def pins(arch):
    cp = arch["copy_paste"]
    paths = [*FILES,*(str(Path(cp[key])/f) for key in ("new_database","donor_database")
                      for f in ("manifest.json","provenance.json")),arch["dataset"]["pointnet"]["encoder"]]
    return {p:sha256(p) for p in paths}


class Trainer(E1d):
    def __init__(self, arch, data, output, resume=None):
        super().__init__(arch,data,output)
        self.donors = CombinedDonors(self.dataset.database,arch["copy_paste"]["new_database"])
        self.identity.update(mode="E.1e", verified_inputs=pins(arch))
        self.identity["code"].update({p:sha256(p) for p in FILES})
        self.tb = None
        if resume:
            self.load(resume)
            self.stats_epoch = self.next_epoch
            if self.partial:
                from lib.utils.raw_point_copy_paste_paper_e1d import tally
                for stats in self.partial["meters"]["cp"]:
                    tally(self.counts, stats)
                    extra_tally(self.counts,stats)

    def prepare(self, sample):
        if self.enabled:
            with deterministic_encoder():
                augment(sample,self.donors,self.data["learning_map"],self.arch["copy_paste"],self.device)

    def update(self,batch):
        out = super().update(batch)
        for s in batch:
            extra_tally(self.counts,s["stats"])
        return out

    def save(self,epoch,next_batch=0,partial=None,best=False,elapsed_seconds=None):
        rec = dict(epoch=epoch,**summary(self.counts))
        atomic_json(self.output/"guided_cp_progress.json",dict(rec,next_batch=next_batch))
        if partial is None and not best:
            path = self.output/"guided_cp_epochs.json"
            history = json.loads(path.read_text()) if path.exists() else {}
            history[str(epoch)] = rec
            atomic_json(path,history)
            # Derive CSV from the epoch-keyed history so replaying an epoch is idempotent.
            path = self.output/"donor_stats.csv"
            temp = path.with_suffix(".tmp")
            with temp.open("w",newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["epoch","learning_id","selected","pasted","rejections","attempted_scans",
                                 "rate_1plus","rate_3","objects_per_attempt"])
                for row in sorted(history.values(),key=lambda r:r["epoch"]):
                    for c in CLASSES:
                        prefix = f"failed_{c}:"
                        reasons = {k[len(prefix):]:v for k,v in row.items() if k.startswith(prefix)}
                        writer.writerow([row["epoch"],c,row[f"selected_{c}"],row[f"class_{c}"],json.dumps(reasons),
                                         row["attempted_scans"],row["success_rate"],row["rate_3"],row["objects_per_attempt"]])
            temp.replace(path)
            if self.tb:
                r = self.history[-1]
                for k,tag in (("training_loss","training/loss"),("training_acc","training/acc"),
                              ("training_iou","training/mIoU"),("lr","training/learning_rate"),
                              ("miou","validating/mIoU"),("loss","validating/loss"),("accuracy","validating/acc"),
                              ("best_miou","validation/best_mIoU")):
                    self.tb.add_scalar(tag,r[k],epoch)
                for name,v in r["class_iou"].items():
                    self.tb.add_scalar("validation/class_iou/"+name,v,epoch)
                for k in ("rate_3","objects_per_attempt","success_rate"):
                    self.tb.add_scalar("paper_cp/"+k,rec[k],epoch)
                self.tb.flush()
        super().save(epoch,next_batch,partial,best,elapsed_seconds)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    gate=json.loads((PREFLIGHT/"verification.json").read_text())
    arch,data=config(gate["budget"])
    if gate["status"]!="PASS" or gate["inputs"]!=pins(arch):
        raise RuntimeError("Preflight or hash check failed")
    if args.resume and not (RUN/"latest.path").is_file():
        raise FileNotFoundError("No E.1e checkpoint")
    RUN.mkdir(parents=True,exist_ok=args.resume)
    logger.add(RUN/"train_log.txt",mode="a")
    trainer=Trainer(arch,data,RUN,RUN/"latest.path" if args.resume else None)
    if not args.resume:
        for p in trainer.identity["code"]:
            dest=RUN/"source"/p
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(p,dest)
        (RUN/"training_config.yaml").write_text(yaml.safe_dump(arch,sort_keys=False))
        shutil.copyfile(LABELS,RUN/"dataset_config.yaml")
        shutil.copyfile(PREFLIGHT/"verification.json",RUN/"verified_preflight.json")
        atomic_json(RUN/"metadata.json",dict(identity=trainer.identity,seed=0,scratch=True,
                    commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                    branch=subprocess.check_output(["git","branch","--show-current"],text=True).strip(),
                    torch=str(torch.__version__),gpu=torch.cuda.get_device_name()))
        atomic_json(RUN/"results.json",dict(status="running",completed_epoch=-1))
        atomic_json(RUN/"training_history.json",[])
    logger.info(f"E.1e PID={os.getpid()} | config={CONFIG} | budget={gate['budget']} | "
                f"donors={dict((c,len(g)) for c,g in trainer.donors.groups.items())} | resume={args.resume}")
    with SummaryWriter(str(RUN/"tb_view/events")) as trainer.tb:
        trainer.train()


if __name__=="__main__":
    main()
