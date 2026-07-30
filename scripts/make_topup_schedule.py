#!/usr/bin/env python3
"""Rebuild the val_split=0.1 cyclic top-up schedule without dino_classification.

Reimplements the two functions that matter -- stratified_train_val_test's nested split and
syke-pic's oversample_class traversal -- so a pod can produce the identical schedule from
ifcb_records.csv alone. Verified against the dino splitter's output.

Both details are load-bearing:
  * the split is NESTED: test_size is carved first, then val_size/((1-test_size)+val_size)
    OF THE REMAINDER, and random_state is 42 (load_and_split_dataset never forwards its seed)
  * classes with fewer than 3 rows in train+val stay wholly in train
"""
import json, sys
import pandas as pd
from sklearn.model_selection import train_test_split

rec_csv, out_path = sys.argv[1], sys.argv[2]
rec = pd.read_csv(rec_csv)
rec = rec[rec["Folder"] != "Unclassifiable"]
rec["_strat"] = rec["Class"].fillna("Unknown")

train_val, _test = train_test_split(rec, test_size=0.2, random_state=42,
                                    stratify=rec["_strat"], shuffle=True)
val_size, test_size = 0.1, 0.2
train_split = 1 - test_size
tr_rows = []
for folder, g in train_val.groupby("Folder"):
    if len(g) < 3:
        tr_rows.append(g); continue
    g = g.sample(frac=1.0, random_state=42)
    n_val = max(1, int(round(len(g) * val_size / (train_split + val_size))))
    tr_rows.append(g.iloc[n_val:])
train = pd.concat(tr_rows)

sched = {}
for folder, g in train.groupby("Folder"):
    files = sorted(g["image"])
    if len(files) >= 100:
        continue
    out, i = [], 0
    while len(files) + len(out) < 100:
        out.append(files[i]); i = (i + 1) % len(files)
    sched[folder] = out
json.dump({"schedule": sched}, open(out_path, "w"))
print(f"{len(sched)} classes, {sum(map(len, sched.values()))} slots -> {out_path}")
