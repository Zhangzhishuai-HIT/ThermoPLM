"""ESM-2 protein representations for ThermoPLM.

Sequences are reduced to the 20 standard amino acids (other residues become X; stop and gap
characters are removed). Each protein is represented by the mean of the final-layer residue
representations of ESM-2 (650M parameters). Proteins longer than the 1,022-residue limit are
covered by overlapping windows of 1,022 residues at a stride of 512 (the last window ends at the
C-terminus); residue representations are summed across windows and divided by the number of
windows covering each residue before averaging, which for shorter proteins is the ordinary mean.

Output: <out>/chunk_NNNN.npy (float16, proteins x 1280) and <out>/chunk_NNNN.ids (one protein
identifier per line, the first word of the FASTA header). Finished chunks are skipped, so a run
can be resumed; --shard i/n lets n processes share one output directory.

Requires a CUDA GPU and the fair-esm package; --ckpt is the esm2_t33_650M_UR50D.pt checkpoint.

Usage
  python embed.py --fasta proteins.faa[.gz] --out emb_esm2/ --ckpt esm2_t33_650M_UR50D.pt
"""
import argparse
import gzip
import os
import time

import numpy as np
import torch

STD = set("ACDEFGHIKLMNPQRSTVWY")
TRUNC = 1022          # ESM-2 position-embedding limit


def clean(seq):
    """Strip stops/gaps and map anything outside the 20 standard residues to X."""
    seq = seq.upper().replace("*", "").replace(".", "").replace("-", "")
    return "".join(c if c in STD else "X" for c in seq)


def windows(L, size=TRUNC, stride=512):
    """Start offsets of the windows covering a sequence of length L (the last one ends at L)."""
    if L <= size:
        return [0]
    st = list(range(0, L - size + 1, stride))
    if st[-1] != L - size:
        st.append(L - size)
    return st


def read_fasta(path):
    op = gzip.open if path.endswith(".gz") else open
    name, buf = None, []
    with op(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if name:
                    yield name, clean("".join(buf))
                name, buf = line[1:].split()[0], []
            else:
                buf.append(line.strip())
    if name:
        yield name, clean("".join(buf))


def batches(recs, max_tokens):
    """Length-sorted batching under a token budget (padding-aware)."""
    recs = sorted(recs, key=lambda x: -len(x[1]))
    cur, curmax = [], 0
    for r in recs:
        L = len(r[1]) + 2
        m = max(curmax, L)
        if cur and m * (len(cur) + 1) > max_tokens:
            yield cur
            cur, curmax = [r], L
        else:
            cur.append(r)
            curmax = m
    if cur:
        yield cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True, help="esm2_t33_650M_UR50D.pt")
    ap.add_argument("--max-tokens", type=int, default=12288, help="token budget per forward pass")
    ap.add_argument("--chunk", type=int, default=2000, help="proteins per output file")
    ap.add_argument("--stride", type=int, default=512, help="window stride for long proteins")
    ap.add_argument("--shard", default="0/1", help="i/n: handle only chunks with index %% n == i")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    recs = [(n, s) for n, s in read_fasta(a.fasta) if s]
    print(f"{len(recs)} proteins, {sum(1 for _, s in recs if len(s) > TRUNC)} longer than {TRUNC}",
          flush=True)
    chunks = [recs[i:i + a.chunk] for i in range(0, len(recs), a.chunk)]
    si, _, sn = a.shard.partition("/")
    si, sn = int(si), int(sn or 1)
    todo = [i for i in range(len(chunks)) if i % sn == si
            and not os.path.exists(os.path.join(a.out, f"chunk_{i:04d}.npy"))]
    print(f"{len(chunks)} chunks; shard {si}/{sn}: {len(todo)} to do", flush=True)
    if not todo:
        return

    import esm      # imported here so that embed_prott5.py can reuse this file without fair-esm
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(a.ckpt)
    model = model.eval().cuda().half()
    bc = alphabet.get_batch_converter()
    nlayer = model.num_layers
    pad = alphabet.padding_idx

    t0, done = time.time(), 0
    for ci in todo:
        ck = chunks[ci]
        order = {n: k for k, (n, _) in enumerate(ck)}
        pooled = np.zeros((len(ck), model.embed_dim), dtype=np.float16)
        short = [r for r in ck if len(r[1]) <= TRUNC]
        long_ = [r for r in ck if len(r[1]) > TRUNC]
        for b in batches(short, a.max_tokens):
            _, _, toks = bc(b)
            toks = toks.cuda()
            with torch.no_grad():
                rep = model(toks, repr_layers=[nlayer], return_contacts=False)["representations"][nlayer]
            mask = (toks != pad) & (toks != alphabet.cls_idx) & (toks != alphabet.eos_idx)
            m = mask.unsqueeze(-1)
            p = ((rep * m).sum(1) / m.sum(1)).float().cpu().numpy().astype(np.float16)
            for k, (n, s) in enumerate(b):
                pooled[order[n]] = p[k]
        for n, s in long_:
            L = len(s)
            acc = torch.zeros(L, model.embed_dim, dtype=torch.float32, device="cuda")
            cov = torch.zeros(L, 1, dtype=torch.float32, device="cuda")
            starts = windows(L, TRUNC, a.stride)
            per = max(1, a.max_tokens // (TRUNC + 2))
            for i in range(0, len(starts), per):
                grp = starts[i:i + per]
                _, _, toks = bc([(f"{n}:{st}", s[st:st + TRUNC]) for st in grp])
                with torch.no_grad():
                    rep = model(toks.cuda(), repr_layers=[nlayer],
                                return_contacts=False)["representations"][nlayer]
                for k, st in enumerate(grp):
                    acc[st:st + TRUNC] += rep[k, 1:TRUNC + 1].float()
                    cov[st:st + TRUNC] += 1
                del rep
            pooled[order[n]] = (acc / cov).mean(0).cpu().numpy().astype(np.float16)
            del acc, cov
        np.save(os.path.join(a.out, f"chunk_{ci:04d}.npy"), pooled)
        with open(os.path.join(a.out, f"chunk_{ci:04d}.ids"), "w") as o:
            o.write("\n".join(n for n, _ in ck) + "\n")
        done += len(ck)
        print(f"  chunk {ci:04d}: {done} proteins, {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
