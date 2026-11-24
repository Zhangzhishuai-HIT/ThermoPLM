"""Train the ThermoPLM classifier head on frozen protein language-model representations and
score each protein out of fold.

The classifier has one hidden layer (256 units, ReLU, dropout 0.3) and one sigmoid output per
stress, trained jointly on all stresses with a binary cross-entropy loss weighted by class
frequency (AdamW, learning rate 1e-3, weight decay 1e-4, 60 epochs, batches of 2,048).
Proteins without a label for a stress are treated as unlabelled background, not as negatives.
Features are standardised with the mean and standard deviation of the training folds.

Cross-validation uses the five folds of one partition (column fold_seed<seed> of the label
table), so each protein is scored by a head that did not see it, nor any protein of its
homology cluster when the folds are drawn over clusters.

Inputs
  --labels  tab-separated table with columns protein, is_label (1/0), stresses (comma-separated,
            empty for unlabelled proteins) and fold_seed0 ... fold_seed4 (fold 0-4)
  --emb     directory of chunk_NNNN.npy / chunk_NNNN.ids written by embed.py or embed_prott5.py
Output
  --out     tab-separated table: protein, score_<stress> for every stress (out-of-fold sigmoids)

Usage
  python train.py --labels labels.tsv --emb emb_esm2/ --seed 0 --out cv_esm2_seed0.tsv
"""
import argparse
import csv
import os
import time

import numpy as np

STRESSES = ["heat", "drought", "salt", "cold", "osmotic", "metal", "hypoxia"]


def load_emb(d):
    """Embeddings of a directory written by embed.py / embed_prott5.py: (ids, float32 matrix)."""
    ids, mats = [], []
    for f in sorted(os.listdir(d)):
        if f.endswith(".npy") and f.startswith("chunk_"):
            mats.append(np.load(os.path.join(d, f)))
            with open(os.path.join(d, f[:-4] + ".ids")) as fh:
                ids += [l.strip() for l in fh if l.strip()]
    X = np.concatenate(mats).astype(np.float32)
    assert len(ids) == X.shape[0], f"{len(ids)} ids vs {X.shape[0]} rows"
    return ids, X


def load_labels(path, ids):
    """Rows of the label table that have an embedding, and their label matrix (rows x stresses)."""
    row = {p: i for i, p in enumerate(ids)}
    rows = [r for r in csv.DictReader(open(path), delimiter="\t") if r["protein"] in row]
    idx = np.array([row[r["protein"]] for r in rows])
    Y = np.zeros((len(rows), len(STRESSES)), dtype=np.int8)
    for i, r in enumerate(rows):
        if r["is_label"] == "1":
            for s in r["stresses"].split(","):
                if s in STRESSES:
                    Y[i, STRESSES.index(s)] = 1
    return rows, idx, Y


def train_mlp(Xtr, Ytr, Wtr, Xte, seed, epochs=60, hidden=256, lr=1e-3, wd=1e-4,
              dropout=0.3, return_net=False):
    """Fit the head on (Xtr, Ytr) with positive-class weights Wtr and return sigmoids for Xte."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    dev = "cpu"
    xt = torch.tensor(Xtr, device=dev)
    yt = torch.tensor(Ytr, dtype=torch.float32, device=dev)
    pw = torch.tensor(Wtr, dtype=torch.float32, device=dev)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], hidden), nn.ReLU(), nn.Dropout(dropout),
                        nn.Linear(hidden, Ytr.shape[1])).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    n = xt.shape[0]
    bs = 2048
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(net(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        pred = torch.sigmoid(net(torch.tensor(Xte, device=dev))).numpy()
    return (pred, net) if return_net else pred


def class_weights(Y):
    """Positive-class weight of each stress: unlabelled-or-negative count over positive count."""
    return np.array([(len(Y) - Y[:, k].sum()) / max(Y[:, k].sum(), 1) for k in range(Y.shape[1])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--seed", type=int, default=0, help="partition: uses column fold_seed<seed>")
    ap.add_argument("--init-seed", type=int, default=None,
                    help="seed of the weight initialisation (default: --seed)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    init = a.seed if a.init_seed is None else a.init_seed

    ids, X = load_emb(a.emb)
    print(f"embeddings: {X.shape[0]} proteins x {X.shape[1]} dims", flush=True)
    rows, idx, Y = load_labels(a.labels, ids)
    fold = np.array([int(r[f"fold_seed{a.seed}"]) for r in rows])
    print(f"{len(rows)} labelled-table rows with an embedding; positives "
          f"{ {s: int(Y[:, k].sum()) for k, s in enumerate(STRESSES)} }", flush=True)

    P = np.zeros((len(rows), len(STRESSES)))
    t0 = time.time()
    for f in range(5):
        te = np.flatnonzero(fold == f)
        tr = np.flatnonzero(fold != f)
        Xtr, Xte = X[idx[tr]], X[idx[te]]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr = (Xtr - mu) / sd
        Xte = (Xte - mu) / sd
        Ytr = Y[tr]
        P[te] = train_mlp(Xtr, Ytr, class_weights(Ytr), Xte, init)
        print(f"  fold {f}: train {len(tr)}, test {len(te)} ({time.time() - t0:.0f} s)", flush=True)

    with open(a.out, "w") as o:
        o.write("protein\t" + "\t".join(f"score_{s}" for s in STRESSES) + "\n")
        for i, r in enumerate(rows):
            o.write(r["protein"] + "\t" + "\t".join(f"{P[i, k]:.6f}" for k in range(len(STRESSES)))
                    + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
