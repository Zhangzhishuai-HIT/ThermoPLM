# ThermoPLM

ThermoPLM scores plant proteins for involvement in heat tolerance from their amino-acid sequence. Two classifiers of identical architecture are trained on frozen representations of two protein language models, ESM-2 (650M parameters) and ProtT5-XL, and the ThermoPLM score of a protein is the average of the normalised ranks that the two classifiers assign to it.

This repository contains the model code: protein embedding, classifier training, scoring and rank fusion.

## Requirements

- Python 3.9 or later with numpy, scipy and PyTorch
- [fair-esm](https://github.com/facebookresearch/esm) and the `esm2_t33_650M_UR50D.pt` checkpoint, for ESM-2
- transformers, sentencepiece and a local copy of `Rostlab/prot_t5_xl_uniref50`, for ProtT5-XL
- a CUDA GPU for the embedding step; training and scoring run on CPU

## Files

| Script | What it does |
|---|---|
| `embed.py` | ESM-2 representation of each protein: final-layer residue representations averaged over the protein; proteins longer than 1,022 residues are covered by overlapping windows (stride 512) averaged by coverage |
| `embed_prott5.py` | ProtT5-XL representation with the same sequence cleaning and windows |
| `train.py` | classifier head (one hidden layer of 256 units, one sigmoid output per stress, class-weighted binary cross-entropy) trained by five-fold cross-validation; writes out-of-fold scores |
| `final_ensemble.py` | heads refitted on all labelled proteins, one per initialisation seed |
| `score.py` | scores embedded proteins with one or several heads (several are averaged) |
| `fuse.py` | averages the normalised ranks of the ESM-2 and ProtT5 scores into the ThermoPLM score |

## Usage

1. Embed the proteins with both language models:

   ```
   python embed.py --fasta proteins.faa --out emb_esm2/ --ckpt esm2_t33_650M_UR50D.pt
   python embed_prott5.py --fasta proteins.faa --out emb_prott5/ --model prot_t5_xl_uniref50/
   ```

2. Prepare a label table (tab-separated) with the columns `protein` (FASTA identifier), `is_label` (1 for a labelled protein, otherwise 0), `stresses` (comma-separated among heat, drought, salt, cold, osmotic, metal and hypoxia; empty for unlabelled proteins) and `fold_seed0` to `fold_seed4` (fold 0 to 4 of each of five partitions, drawn over homology clusters). Proteins without a label are treated as unlabelled background, not as negatives.

3. Cross-validate each language model, and fit the heads used for scoring:

   ```
   python train.py --labels labels.tsv --emb emb_esm2/ --seed 0 --out cv_esm2_seed0.tsv
   python final_ensemble.py --labels labels.tsv --emb emb_esm2/ --out-dir models/ --name esm2
   python final_ensemble.py --labels labels.tsv --emb emb_prott5/ --out-dir models/ --name prott5
   ```

4. Score new proteins with each language model, then combine the two rankings:

   ```
   python score.py --emb emb_esm2_new/ --model models/esm2_seed0_final.pt,models/esm2_seed1_final.pt,models/esm2_seed2_final.pt,models/esm2_seed3_final.pt,models/esm2_seed4_final.pt --out scores_esm2.tsv
   python score.py --emb emb_prott5_new/ --model models/prott5_seed0_final.pt,models/prott5_seed1_final.pt,models/prott5_seed2_final.pt,models/prott5_seed3_final.pt,models/prott5_seed4_final.pt --out scores_prott5.tsv
   python fuse.py --a scores_esm2.tsv --b scores_prott5.tsv --out scores_thermoplm.tsv
   ```

The ThermoPLM score is a rank percentile within the scored set (1 = top of the ranking), not a probability.

## Licence

MIT. See `LICENSE`.
