"""ProtT5-XL protein representations for ThermoPLM.

Uses the same sequence cleaning and the same overlapping 1,022-residue windows at a stride of
512 as embed.py, so that both language models see identical input. Each protein is represented
by the mean of the final-layer encoder states of ProtT5-XL-UniRef50 over its residues (the
end-of-sequence token is excluded); for long proteins, residue states are averaged by window
coverage before the mean.

Output: <out>/chunk_NNNN.npy (float16, proteins x 1024) and <out>/chunk_NNNN.ids, in the layout
of embed.py. Requires a CUDA GPU and the transformers package; --model is a local copy of
Rostlab/prot_t5_xl_uniref50.

Usage
  python embed_prott5.py --fasta proteins.faa[.gz] --out emb_prott5/ --model prot_t5_xl_uniref50/
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embed import TRUNC, read_fasta, windows      # noqa: E402  same cleaning and windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True, help="local directory of prot_t5_xl_uniref50")
    ap.add_argument("--chunk", type=int, default=2000, help="proteins per output file")
    ap.add_argument("--max-tokens", type=int, default=8192, help="token budget per forward pass")
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

    from transformers import T5EncoderModel, T5Tokenizer
    tok = T5Tokenizer.from_pretrained(a.model, do_lower_case=False, legacy=True)
    model = T5EncoderModel.from_pretrained(a.model).eval().cuda().half()
    dim = model.config.d_model

    def encode(seqs):
        """Final-layer states of the residues of each sequence (padding and EOS removed)."""
        enc = tok.batch_encode_plus([" ".join(s) for s in seqs], add_special_tokens=True,
                                    padding="longest", return_tensors="pt")
        with torch.no_grad():
            h = model(input_ids=enc["input_ids"].cuda(),
                      attention_mask=enc["attention_mask"].cuda()).last_hidden_state
        return [h[k, :len(s)].float() for k, s in enumerate(seqs)]

    t0, done = time.time(), 0
    for ci in todo:
        ck = chunks[ci]
        order = {n: k for k, (n, _) in enumerate(ck)}
        pooled = np.zeros((len(ck), dim), dtype=np.float16)
        short = sorted([(n, s) for n, s in ck if len(s) <= TRUNC], key=lambda x: -len(x[1]))
        long_ = [(n, s) for n, s in ck if len(s) > TRUNC]
        i = 0
        while i < len(short):
            grp = short[i:i + max(1, a.max_tokens // (len(short[i][1]) + 1))]
            for (n, s), h in zip(grp, encode([s for _, s in grp])):
                pooled[order[n]] = h.mean(0).cpu().numpy().astype(np.float16)
            i += len(grp)
        for n, s in long_:
            L = len(s)
            acc = torch.zeros(L, dim, dtype=torch.float32, device="cuda")
            cov = torch.zeros(L, 1, dtype=torch.float32, device="cuda")
            starts = windows(L, TRUNC, a.stride)
            per = max(1, a.max_tokens // (TRUNC + 1))
            for j in range(0, len(starts), per):
                grp = starts[j:j + per]
                for st, h in zip(grp, encode([s[st:st + TRUNC] for st in grp])):
                    acc[st:st + TRUNC] += h
                    cov[st:st + TRUNC] += 1
            pooled[order[n]] = (acc / cov).mean(0).cpu().numpy().astype(np.float16)
            del acc, cov
        np.save(os.path.join(a.out, f"chunk_{ci:04d}.npy"), pooled)
        with open(os.path.join(a.out, f"chunk_{ci:04d}.ids"), "w") as o:
            o.write("\n".join(n for n, _ in ck) + "\n")
        done += len(ck)
        print(f"  chunk {ci:04d}: {done} proteins, {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
