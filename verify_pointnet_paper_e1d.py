"""Preflight only: never initialize formal weights from this diagnostic process."""
from collections import Counter
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from train_pointnet_paper_raw_copy_paste_e1d import (
    Trainer, PREFLIGHT, FILES, read_config, atomic_json, deterministic_encoder, sha256)
from compare_pointnet_paper_e1d import NAMES, locate
from trainer_PointNet_RawCopyPaste_E1 import state_digest
from lib.utils.raw_point_copy_paste_e0b import regenerate_hybrid, rigid_place
from lib.utils.raw_point_copy_paste_paper_e1d import grid, tally, epoch_stats
import numpy as np
import torch


def verify_example(trainer, sample, raw, packed):
    cfg = trainer.arch["copy_paste"]["paper"]
    mapping = trainer.data["learning_map"]
    device = trainer.device
    after, labels, mask = sample["points"], sample["packed"], sample["paste_points"]
    assert len(after) == len(labels) == len(mask)
    np.testing.assert_array_equal(after[:len(raw)],raw)
    np.testing.assert_array_equal(labels[:len(raw)],packed)
    np.testing.assert_array_equal(mask,np.arange(len(after))>=len(raw))
    with deterministic_encoder():
        original = regenerate_hybrid(raw, packed, trainer.encoder, mapping)
        final = regenerate_hybrid(after, labels, trainer.encoder, mapping, mask)
        from verify_raw_copy_paste_e0c import canonical_reference, compare
        independent = canonical_reference(after, labels, mapping, trainer.encoder, device)
        compare(final, independent)
        target = original["label"].numpy().ravel().copy()
        protected = np.isin(target,cfg["protected"])
        np.testing.assert_array_equal(target[protected],final["label"].numpy().ravel()[protected])
        assert not final["paste_mask"].numpy().ravel()[protected].any()
        reserved = np.zeros(target.shape, dtype=bool)
        rows = {r["donor_id"]:r for r in trainer.dataset.database.rows}
        offset = len(raw)
        ids_taken = set(np.unique(packed >> 16))
        for placement in sample["placement"]:
            cid = placement["learning_id"]
            donor = trainer.dataset.database.load(rows[placement["donor_id"]])
            moved,_ = rigid_place(donor[:,:3],placement["yaw"],np.asarray(placement["translation"]))
            all_points = np.column_stack((moved,donor[:,3]))
            cells, inv, valid = grid(all_points,device)
            assert valid.all()
            foreground = protected | reserved
            context = np.isin(target[cells],cfg["allowed_background"][cid]).mean()
            blocked = foreground[cells]
            assert context == placement["context"] and context > .1
            assert blocked.mean() == placement["overlap"] <= cfg["max_overlap"][cid]
            assert (~blocked).mean() == placement["visibility"]
            keep = ~blocked[inv]
            end = offset + keep.sum()
            instance = placement["synthetic_instance_id"]
            assert instance not in ids_taken
            ids_taken.add(instance)
            np.testing.assert_array_equal(after[offset:end],all_points[keep])
            np.testing.assert_array_equal(labels[offset:end],donor[keep,4].astype(np.uint32) | np.uint32(instance << 16))
            retained = cells[~blocked]
            assert not reserved[retained].any()
            reserved[retained] = True
            target[retained] = cid
            offset = end
        assert offset == len(after)
        np.testing.assert_array_equal(reserved,final["paste_mask"].numpy().ravel().astype(bool))
        np.testing.assert_array_equal((original["mask"].numpy().ravel()>0)|reserved,final["mask"].numpy().ravel()>0)
        assert torch.isfinite(final["hybrid24"]).all()
        assert final["point16"].shape == (16,512,512)
        torch.testing.assert_close(final["hybrid24"][7:23],final["point16"].half().float(),rtol=0,atol=0)
        torch.testing.assert_close(final["hybrid24"][23:],final["mask"],rtol=0,atol=0)
        np.savez_compressed(PREFLIGHT/"before_after.npz", scan=sample["scan_id"], before_points=raw,
            before_packed=packed, after_points=after, after_packed=labels, donor_points_mask=mask,
            before_bev_labels=original["label"].numpy(), after_bev_labels=final["label"].numpy(),
            before_obs=original["mask"].numpy(),after_obs=final["mask"].numpy(),
            before_point16=original["hybrid24"][7:23].half().numpy(), after_point16=final["hybrid24"][7:23].half().numpy(),
            paste_mask=final["paste_mask"].numpy())
    atomic_json(PREFLIGHT/"example.json", dict(scan=sample["scan_id"],placements=sample["placement"],stats=sample["stats"]))


def main():
    PREFLIGHT.mkdir(exist_ok=True)
    if (PREFLIGHT / "verification.json").exists():
        raise FileExistsError("Do not overwrite a committed preflight")
    suites = ["test_pointnet_paper_e1d.py","test_basic_raw_copy_paste_e1c.py","test_pointnet_training_e1.py"]
    for suite in suites:
        with (PREFLIGHT/(suite+".log")).open("w") as stream:
            subprocess.run([sys.executable,"-B","-m","unittest","discover","-s","tests","-p",suite,"-v"],
                           stdout=stream,stderr=subprocess.STDOUT,check=True)
    protected = {str(path):sha256(path) for name in NAMES[:3] for path in locate(name).rglob("*") if path.is_file()}
    arch,data=read_config()
    trainer=Trainer(arch,data,PREFLIGHT)
    frozen=trainer.encoder_state_hash
    counts, examples = Counter(), []
    example=None
    for i in range(400):
        idx=i*347 % len(trainer.dataset)
        sample=trainer.dataset[idx]
        raw, packed=sample["points"],sample["packed"]
        trainer.prepare(sample)
        tally(counts,sample["stats"])
        if not sample["placement"]:
            continue
        if len(examples)<3:
            examples.append(idx)
        if example is None and any(p["blocked_cells"] for p in sample["placement"]):
            verify_example(trainer,sample,raw,packed)
            example=sample["scan_id"]
        if i>=63 and example and len(examples)==3 and all(counts[f"class_{j}"] for j in range(2,7)):
            break
    assert example is not None and len(examples)==3
    assert all(counts[f"class_{j}"] for j in range(2,7))
    from lib.dataset.SemanticKitti_PointNet_RawCopyPaste_E1 import SemanticKitti
    old_off=SemanticKitti("SemanticKitti",data,arch["copy_paste"],False)
    trainer.enabled=False
    a,b=trainer.dataset[100],old_off[100]
    trainer.prepare(a)
    np.testing.assert_array_equal(a["points"],b["points"])
    np.testing.assert_array_equal(a["packed"],b["packed"])
    assert a["augmentation_seed"]==b["augmentation_seed"] and a["paste_points"] is None
    trainer.enabled=True
    with patch("lib.utils.raw_point_copy_paste_e0b.visibility_filter",side_effect=AssertionError), \
         patch("lib.utils.raw_point_copy_paste_e0b.raw_copy_paste",side_effect=AssertionError):
        loss,_,_,diagnostics=trainer.update([trainer.dataset[i] for i in examples])
    assert diagnostics["input_shape"]==[3,24,512,512] and diagnostics["gradient_norm"]>0
    assert state_digest(trainer.model.state_dict())!=trainer.initial_hash
    assert frozen==state_digest(trainer.encoder.state_dict())
    assert all(not p.requires_grad and p.grad is None for p in trainer.encoder.parameters())
    assert all(sha256(path)==digest for path,digest in protected.items())
    atomic_json(PREFLIGHT/"verification.json",dict(status="PASS",inputs={p:sha256(p) for p in FILES},
                regression_suites=suites,protected_prior_artifacts=protected,real_scan_stats=epoch_stats(counts),
                sample=example,loss=loss,diagnostics=diagnostics,frozen_encoder_hash=frozen,
                initial_state_sha256=trainer.initial_hash))
    print("PASS:",example,loss,diagnostics["input_shape"],epoch_stats(counts),flush=True)


if __name__=="__main__":
    main()
