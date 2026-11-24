"""Fit the deployed ThermoPLM heads: one head per initialisation seed on all labelled data.

The scorer applied to new proteins averages several heads that differ only in weight
initialisation, because the membership of a top-ranked list is the least stable output of a
single head. These refits use every label, so they have no held-out set; accuracy is measured
by the cross-validation in train.py.

Each checkpoint stores the head's weights together with the standardisation (mean and standard
deviation of the training features) that score.py applies before scoring.

Usage
  python final_ensemble.py --labels labels.tsv --emb emb_esm2/ --out-dir models/ --name esm2
      [--seeds 0,1,2,3,4]
writes models/esm2_seed<k>_final.pt for every seed.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import STRESSES, class_weights, load_emb, load_labels, train_mlp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]

    import torch

    ids, X = load_emb(a.emb)
    rows, idx, Y = load_labels(a.labels, ids)
    print(f"{len(rows)} rows, positives { {s: int(Y[:, k].sum()) for k, s in enumerate(STRESSES)} }",
          flush=True)
    Xall = X[idx]
    mu, sd = Xall.mean(0), Xall.std(0) + 1e-6
    Xn = (Xall - mu) / sd
    pw = class_weights(Y)

    os.makedirs(a.out_dir, exist_ok=True)
    for s in seeds:
        dst = os.path.join(a.out_dir, f"{a.name}_seed{s}_final.pt")
        _, net = train_mlp(Xn, Y, pw, Xn[:1], s, return_net=True)
        torch.save({"state_dict": net.state_dict(), "mu": mu, "sd": sd,
                    "outputs": STRESSES, "hidden": 256, "in_dim": Xall.shape[1]}, dst)
        print(f"seed {s}: wrote {dst}", flush=True)


if __name__ == "__main__":
    main()
