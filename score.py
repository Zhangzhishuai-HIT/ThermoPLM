"""Score proteins with trained ThermoPLM heads.

Reads the embeddings written by embed.py or embed_prott5.py, applies one or several
checkpoints of final_ensemble.py (several are averaged; each carries its own feature
standardisation) and writes one row per protein. Chunks are scored one at a time, so large
proteome sets do not need to fit in memory.

Protein identifiers of the form <accession>|<protein> are split into two columns; plain
identifiers are written to both.

Usage
  python score.py --emb emb_esm2/ --model models/esm2_seed0_final.pt,...,models/esm2_seed4_final.pt
      --out scores_esm2.tsv
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn


def chunk_files(d):
    out = []
    for f in sorted(os.listdir(d)):
        if f.startswith("chunk_") and f.endswith(".npy"):
            ids = os.path.join(d, f[:-4] + ".ids")
            if os.path.exists(ids):
                out.append((os.path.join(d, f), ids))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--model", required=True, help="one checkpoint, or several separated by commas")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    paths = [p for p in a.model.split(",") if p]
    nets, mus, sds, outs = [], [], [], None
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        if outs is None:
            outs = ck["outputs"]
        elif ck["outputs"] != outs:
            raise SystemExit(f"{p} predicts {ck['outputs']}, not {outs}")
        net = nn.Sequential(nn.Linear(ck["in_dim"], ck["hidden"]), nn.ReLU(), nn.Dropout(0.0),
                            nn.Linear(ck["hidden"], len(outs)))
        net.load_state_dict(ck["state_dict"])
        net.eval()
        nets.append(net)
        mus.append(ck["mu"])
        sds.append(ck["sd"])
    print(f"{len(nets)} checkpoint(s)", flush=True)

    cf = chunk_files(a.emb)
    print(f"{len(cf)} embedding chunks in {a.emb}", flush=True)
    n = 0
    with open(a.out, "w") as o:
        o.write("accession\tprotein\t" + "\t".join(f"score_{s}" for s in outs) + "\n")
        for npy, idf in cf:
            X = np.load(npy).astype(np.float32)
            ids = [l.strip() for l in open(idf) if l.strip()]
            if len(ids) != X.shape[0]:
                raise SystemExit(f"{npy}: {X.shape[0]} rows but {len(ids)} ids")
            with torch.no_grad():
                P = np.mean([torch.sigmoid(nt(torch.tensor((X - mu) / sd))).numpy()
                             for nt, mu, sd in zip(nets, mus, sds)], axis=0)
            for i, pid in enumerate(ids):
                acc, _, prot = pid.partition("|")
                o.write(f"{acc}\t{prot or acc}\t"
                        + "\t".join(f"{P[i, k]:.6f}" for k in range(len(outs))) + "\n")
            n += len(ids)
            del X, P
    print(f"scored {n} proteins -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
