"""Combine the ESM-2 and ProtT5 scores of the same proteins into the ThermoPLM score.

For each stress, the scores of each model are converted to ranks scaled to [0, 1] (tied scores
share their average rank) and the two scaled ranks are averaged with equal weight. The result is
a rank percentile (1 = top of the ranking), not a probability; a sidecar <out>.meta.json records
this. The combination has no fitted parameters.

Proteins are matched by the `protein` column; only proteins scored by both models are kept.
Identifier columns are taken from the first table; score and rank columns are not carried over.

Usage
  python fuse.py --a scores_esm2.tsv --b scores_prott5.tsv --out scores_thermoplm.tsv
"""
import argparse
import csv
import json

import numpy as np
from scipy.stats import rankdata

STRESSES = ["heat", "drought", "salt", "cold", "osmotic", "metal", "hypoxia"]


def read(path):
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    return {r["protein"]: r for r in rows}, [r["protein"] for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="scores of the first model (e.g. ESM-2)")
    ap.add_argument("--b", required=True, help="scores of the second model (e.g. ProtT5)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stresses", default=",".join(STRESSES))
    a = ap.parse_args()
    ss = [s for s in a.stresses.split(",") if s]

    A, order = read(a.a)
    B, _ = read(a.b)
    both = [p for p in order if p in B]
    print(f"{len(A)} + {len(B)} proteins, {len(both)} scored by both")

    out = {}
    for s in ss:
        col = "score_" + s
        va = np.array([float(A[p][col]) for p in both])
        vb = np.array([float(B[p][col]) for p in both])
        ra = (rankdata(va, method="average") - 1) / (len(va) - 1)
        rb = (rankdata(vb, method="average") - 1) / (len(vb) - 1)
        out[s] = (ra + rb) / 2
    keep = [c for c in A[both[0]] if not c.startswith(("score_", "rank_"))]
    with open(a.out, "w") as o:
        o.write("\t".join(keep) + "\t" + "\t".join("score_" + s for s in ss) + "\n")
        for i, p in enumerate(both):
            o.write("\t".join(A[p][c] for c in keep) + "\t"
                    + "\t".join(f"{out[s][i]:.9f}" for s in ss) + "\n")
    json.dump({"inputs": [a.a, a.b], "stresses": ss, "n_scored": len(both),
               "score": "mean of the two models' ranks scaled to [0, 1]; a rank percentile, "
                        "not a probability"},
              open(a.out + ".meta.json", "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
