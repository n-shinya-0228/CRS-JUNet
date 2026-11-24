"""Train-only, index-preserving E.1e donor extraction. Never synthesizes rider/vehicle pairs."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from lib.utils.raw_point_copy_paste_e0c import TRAIN_SEQUENCES, check_sequence
from lib.utils.pointnet_copy_paste_e0 import sha256

NEW = (7, 8, 18, 19)
DB = Path("copy_paste_database_e1e_new4_trainonly")
LABELS = Path("config/labels/semantic-kitti.yaml")
SETTINGS = dict(static_frame_stride=10, radius_m={7:.45,8:.48,18:.32,19:.28},
                static_note="One in ten frames/sequence to limit duplicate static infrastructure; riders every frame")


def components(xyz, radius):
    if not len(xyz):
        return []
    pairs = cKDTree(xyz).query_pairs(radius, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs),np.uint8), pairs.T), shape=(len(xyz),len(xyz)))
    count, labels = connected_components(graph, directed=False)
    return [np.flatnonzero(labels == k) for k in range(count)]


def candidates(points, packed, mapped, cid):
    idx = np.flatnonzero(mapped == cid)
    if not len(idx):
        return
    xyz = points[idx,:3]
    ids = packed[idx] >> 16
    groups = []
    if cid in (7,8):
        for inst in np.unique(ids[ids != 0]):
            members = np.flatnonzero(ids == inst)
            mixed = np.any(mapped[(packed >> 16) == inst] != cid)
            groups.append((members, "whole_instance", ["mixed_instance"] if mixed else []))
        zero = np.flatnonzero(ids == 0)
        groups.extend((zero[g],"isolated_rider_cluster",[]) for g in components(xyz[zero], SETTINGS["radius_m"][cid]))
    else:
        groups.extend((g,"semantic_cluster",[]) for g in components(xyz, SETTINGS["radius_m"][cid]))
    tree = cKDTree(xyz)
    for group, method, reasons in groups:
        obj = xyz[group].astype(np.float64)
        original = idx[group]
        if method != "whole_instance" and len(np.unique(ids[group])) != 1:
            reasons.append("mixed_instance_ids")
        if len(obj)<(24 if cid in (7,8) else 8):
            reasons.append("sparse_fragment")
        if len(np.unique(packed[original]&65535)) != 1:
            reasons.append("mixed_raw_ids")
        center = obj.mean(0)
        if np.linalg.norm(center[:2]) >= 51.2:
            reasons.append("outside_training_radius")
        delta = obj-center
        _, axes = np.linalg.eigh(delta.T@delta)
        _, xyaxes = np.linalg.eigh(delta[:,:2].T@delta[:,:2])
        xy = np.ptp(delta[:,:2]@xyaxes, axis=0)
        length,width = sorted(map(float,xy),reverse=True)
        height=float(np.ptp(obj[:,2]))
        ext=np.ptp(delta@axes,axis=0)
        shape=dict(length=length,width=width,height=height,pca_extent=ext.tolist(),
                   main_axis_vertical=abs(float(axes[2,-1])),range_m=float(np.linalg.norm(center[:2])))
        if cid in (7,8):
            if not (.7<=height<=2.5 and .65<=length<=3.2 and .12<=width<=1.35):
                reasons.append("rider_extent")
            if not reasons and len(components(obj,SETTINGS["radius_m"][cid]))!=1:
                reasons.append("fragmented_rider")
        elif cid==18:
            if not (1.2<=height<=10 and length<=.48 and height>=4*length and shape["main_axis_vertical"]>=.94):
                reasons.append("not_single_vertical_shaft")
            if not reasons:
                # Reject multiple separated shafts at the same height, not just a skinny global box.
                for band in np.array_split(obj[np.argsort(obj[:,2])],4):
                    if len(band)>=4 and len(components(band[:,:2],.13))>1:
                        reasons.append("multiple_shafts"); break
        else:
            if not (.25<=height<=2.6 and .22<=length<=2.6 and width<=.28 and
                    width<=.35*length and ext[0]<=.23 and abs(axes[2,0])<.45):
                reasons.append("not_single_sign_panel")
            if not reasons:
                # Obvious gaps across a board are evidence of adjacent panels or partial returns.
                for axis in (obj[:,2], delta[:,:2]@xyaxes[:,-1]):
                    if np.diff(np.sort(axis)).max(initial=0)>.23:
                        reasons.append("panel_gap"); break
        if not reasons and method != "whole_instance":
            near=tree.query_ball_point(obj, .75 if cid in (7,8) else .55)
            if any(v not in set_group for set_group in [set(group)] for neighbors in near for v in neighbors):
                reasons.append("ambiguous_nearby_same_class")
        if not reasons and (points[original,3].min()<0 or points[original,3].max()>1):
            reasons.append("invalid_remission")
        clearance=0.
        if cid==19 and not reasons:
            near=(np.abs(points[:,0]-center[0])<2)&(np.abs(points[:,1]-center[1])<2)
            ground=points[near & np.isin(mapped,[9,10,11,12,17])]
            if len(ground)<5:
                reasons.append("unknown_sign_elevation")
            else:
                clearance=float(obj[:,2].min()-np.median(ground[:,2]))
                if not .35<=clearance<=4.0:
                    reasons.append("ambiguous_sign_elevation")
        yield original,method,reasons,shape,clearance


def plot(path, points, idx, cid, reason):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p=points[idx,:3]; center=p.mean(0); p=p-center
    near=points[np.linalg.norm(points[:,:3]-center,axis=1)<max(2.5,np.linalg.norm(p,axis=1).max()+.5),:3]-center
    fig=plt.figure(figsize=(7,3.5))
    for i,az in enumerate((35,125),1):
        ax=fig.add_subplot(1,2,i,projection="3d")
        ax.scatter(*near[::max(1,len(near)//1600)].T,s=1,c="gray",alpha=.15)
        ax.scatter(*p.T,s=4,c=p[:,2],cmap="viridis")
        ax.view_init(18,az); ax.set_box_aspect([1,1,1])
        half=max(1.1,np.abs(p).max()*1.12)
        ax.set(xlim=(-half,half),ylim=(-half,half),zlim=(-half,half))
    fig.suptitle(f"{cid}: {reason} ({len(p)} points)")
    fig.savefig(path,dpi=115);plt.close(fig)


def audit_donor(file,row,source_points,source_labels,lookup):
    if sha256(file)!=row["sha256"]:
        raise ValueError("donor checksum")
    check_sequence(row["source_sequence"])
    with np.load(file,allow_pickle=False) as d:
        idx=d["source_point_indices"]
        assert len(idx)==len(np.unique(idx))==row["point_count"] and np.all(np.diff(idx)>0)
        np.testing.assert_array_equal(d["points"],source_points[idx]);np.testing.assert_array_equal(d["packed_labels"],source_labels[idx])
        assert np.isfinite(d["points"]).all() and np.all(lookup[source_labels[idx]&65535]==row["learning_id"])
        if row["method"]=="whole_instance":
            np.testing.assert_array_equal(idx,np.flatnonzero(source_labels>>16==row["source_instance_id"]))


def main():
    cfg=yaml.safe_load(LABELS.read_text()); assert tuple(f"{s:02d}" for s in cfg["split"]["train"])==TRAIN_SEQUENCES
    lookup=np.full(65536,-1,int)
    for k,v in cfg["learning_map"].items(): lookup[k]=v
    DB.mkdir(exist_ok=False); (DB/"objects").mkdir(); (DB/"examples").mkdir()
    hashes=dict(code_sha256=sha256(__file__),labels_sha256=sha256(LABELS),
                settings_sha256=hashlib.sha256(json.dumps(SETTINGS,sort_keys=True).encode()).hexdigest())
    manifest=[];counts=defaultdict(Counter); dist=defaultdict(list); figures=Counter();scans=0
    with (DB/"candidates.csv").open("x",newline="") as stream:
        writer=csv.writer(stream);writer.writerow(["sequence","frame","cid","first_index","points","method","length","width","height","reasons"])
        for seq in TRAIN_SEQUENCES:
            check_sequence(seq)
            seqdir=Path("SemanticKitti/sequences")/seq
            for path in sorted((seqdir/"labels").glob("*.label")):
                scans+=1
                labels=np.fromfile(path,np.uint32);mapped=lookup[labels&65535]
                if (mapped<0).any(): raise ValueError(path)
                classes=NEW if int(path.stem)%SETTINGS["static_frame_stride"]==0 else NEW[:2]
                if not any((mapped==c).any() for c in classes): continue
                scan=seqdir/"velodyne"/(path.stem+".bin")
                points=np.fromfile(scan,np.float32).reshape(-1,4)
                if len(points)!=len(labels) or not np.isfinite(points).all():raise ValueError(scan)
                src=None
                for cid in classes:
                    for idx,method,reasons,shape,gap in candidates(points,labels,mapped,cid):
                        key="reject" if reasons else "accept"
                        writer.writerow([seq,path.stem,cid,idx[0],len(idx),method,shape["length"],shape["width"],shape["height"],";".join(reasons)])
                        counts[cid][key]+=1;counts[cid].update(reasons)
                        dist[cid,key].append([len(idx),shape["length"],shape["width"],shape["height"]])
                        if figures[cid,key]<3 and len(idx)>=8:
                            fname=f"{cid}_{key}_{figures[cid,key]}_{seq}_{path.stem}_{idx[0]}.png"
                            plot(DB/"examples"/fname,points,idx,cid,",".join(reasons) or "accepted")
                            figures[cid,key]+=1
                        if reasons: continue
                        if src is None: src=dict(bin_sha256=sha256(scan),packed_sha256=sha256(path))
                        donor_id=f"e1e_{seq}_{path.stem}_{idx[0]}"
                        file=DB/"objects"/(donor_id+".npz")
                        obj=points[idx]; packed=labels[idx]
                        np.savez_compressed(file,points=obj,packed_labels=packed,source_point_indices=idx)
                        row=dict(donor_id=donor_id,file=str(file.relative_to(DB)),sha256=sha256(file),
                                 source_sequence=seq,source_frame=path.stem,source_instance_id=int(packed[0]>>16),
                                 raw_ids=np.unique(packed&65535).tolist(),learning_id=cid,point_count=len(idx),
                                 bbox_min=obj[:,:3].min(0).tolist(),bbox_max=obj[:,:3].max(0).tolist(),shape=shape,
                                 method=method,ground_clearance=gap,split="train",reasons=[],source=src,**hashes)
                        audit_donor(file,row,points,labels,lookup);manifest.append(row)
            print(seq,scans,dict((k,v["accept"]) for k,v in counts.items()),flush=True)
    if scans!=19130 or any(not counts[c]["accept"] for c in NEW):raise ValueError(f"incomplete extraction: {counts}")
    (DB/"manifest.json").write_text(json.dumps(manifest))
    (DB/"quality.json").write_text(json.dumps(dict(counts=counts,scans=scans,settings=SETTINGS,
         quantiles={f"{c}_{k}":dict(zip(("points","length","width","height"),
                     np.quantile(v,[0,.25,.5,.75,1],axis=0).T.tolist())) for (c,k),v in dist.items()}),indent=2))
    (DB/"provenance.json").write_text(json.dumps(dict(complete=True,settings=SETTINGS,source_scans=scans,
           manifest_sha256=sha256(DB/"manifest.json"),sequences=TRAIN_SEQUENCES,**hashes),indent=2))


if __name__=="__main__":
    main()
