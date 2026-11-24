"""Complete the historical donor selection counts and export the paired preflight table."""
import csv
import json
from collections import Counter
from pathlib import Path

from compare_pointnet_paper_e1d import NAMES, locate


def main():
    preflight = Path("logs/SJNet3_PointNet_stepE1e_raw_cp_paper3_9class_preflight")
    gate = json.loads((preflight/"verification.json").read_text())
    if gate["status"] != "PASS":
        raise ValueError("Require a completed preflight")
    hist = preflight/"historical_e1d.json"
    baseline = json.loads(hist.read_text())
    # E.1d's paper history contains pasted counts only. Selections live in its other CSV.
    source = locate(NAMES[-1])/"cp_stats.csv"
    selected = Counter()
    with source.open() as stream:
        for row in csv.DictReader(stream):
            selected.update(json.loads(row["class_attempts"]))
    baseline.update({"selected_"+k:v for k,v in selected.items()},selected_source=str(source))
    temp = hist.with_suffix(".tmp")
    temp.write_text(json.dumps(baseline,indent=2))
    temp.replace(hist)
    comparison = json.loads((preflight/gate["paired"]).read_text())
    with (preflight/"paired_donor_stats.csv").open("w",newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id","class","E1d_selected","E1d_pasted","E1e_selected","E1e_pasted"])
        for c,name in ((2,"bicycle"),(3,"motorcycle"),(4,"truck"),(5,"other-vehicle"),(6,"person"),
                       (7,"bicyclist"),(8,"motorcyclist"),(18,"pole"),(19,"traffic-sign")):
            writer.writerow([c,name,*[comparison[method][f"{metric}_{c}"] for method in ("e1d","e1e")
                                       for metric in ("selected","class")]])


if __name__=="__main__":
    main()
