# LM Vocal Conditioning — Diagnosis, Fix, and Experiments

Date: 2026-05-22

## Problem
LM-generated drums were noisy and did not follow the vocals. Inference-time tricks
(temperature, top_k, top_p, energy_gate, mel gating) did not fix it.

## Diagnosis
Conditioning ablation on the old model (exp27/lm13, vq7) — zero each input, measure
val cross-entropy + top-1 token accuracy:

| condition          | CE (bits) | top-1 |
|--------------------|-----------|-------|
| full               | 2.84      | 39.3% |
| zero beat-phase    | 3.16      | 35.0% |
| zero vocal (otz)   | 2.839     | 39.2% |  ← unchanged

Zeroing the vocal changed nothing → the model **completely ignored the vocal stem**.

Root cause: beat-phase enters via `x_cond` (added at every timestep, so unavoidable),
but the vocal only entered via `encoder_kv` cross-attention. With `attn_order=8`
(`[1,2,3,1,2,3,1,2,3,6][d%10]`), only `attn_func=6` attends to `encoder_kv` → just
2 of 20 layers saw the vocal. Under dropout 0.4 + a 5-term composite loss, those 2
layers collapsed to ignoring it.

## Fix
Inject the vocal into `x_cond` per-timestep (same pathway that makes beat-phase work),
lower dropout so the conditioning path can learn, and train plain CE to remove
confounders.

## Experiments (all vq7 unless noted, dropout 0.1, plain CE, data_jungmori)

| exp | model | vocal feature           | val CE | top-1 | vocal effect (zero) |
|-----|-------|-------------------------|--------|-------|---------------------|
| 27  | lm13  | (old) cross-attn only   | 2.84   | 39.3% | ~0.00 bits          |
| 29  | lm15  | others VQ token embed   | 2.69   | 41.5% | 1.32 bits           |
| 30  | lm16  | **vocal MEL (winner)**  | **2.33** | **47.0%** | **6.18 bits**  |
| 31  | lm17  | vocal energy+onset env  | 2.45   | 45.0% | 2.60 bits           |
| 32  | lm18  | vocal mel, but vq8(128) | 3.45   | 35.5% | 7.67 bits           |

- All feature variants beat the old token/cross-attn model.
- Raw vocal **mel** beat the onset/energy envelope and the token embedding.
- **vq8 (128-token codebook) did not help**: even normalized to its larger vocab it
  captured less structure (CE/uniform 0.49 vs vq7 0.47) and top-1 dropped to 35.5%.
  The LM models 32 tokens far more reliably.
- No overfit anywhere (val-train CE gap ≈ 0.04–0.06).

## WINNER: exp30 — vq7 + vocal MEL feature into x_cond
Checkpoint: `ckpt/exp30_ce_best.pkl` (hparams `lm16`).

## How to reproduce / train
```bash
source jd1/bin/activate
python train_lm_ablation.py --cuda 0 --exp_idx 30   # plain CE, no loss flags
```
Validate any LM change with the conditioning ablation: zeroing the vocal MUST raise CE
substantially (exp30: +6.18 bits). If it doesn't, the vocal is being ignored again.

## Code changes

### hparams.py
- MODEL_LIST: added 29→(vq7,lm15), 30→(vq7,lm16), 31→(vq7,lm17), 32→(vq8,lm18).
- New entries lm15 (vocal token embed), lm16 (vocal mel, WINNER), lm17 (vocal envelope),
  lm18 (vq8 + vocal mel). All dropout 0.1, binfo_type dphase, plain CE.

### model/LanguageModel.py
- `make_juke_prior`: `attn_dropout`/`resid_dropout` now follow `args.dropout`
  (were hardcoded 0.3, so lowering dropout was previously a no-op for attn/resid).
- `__init__`: added vocal conditioning paths —
  - `vocal_xcond` → `nn.Embedding(others_codebook_size, d_model)` (token embed; vq7
    others codebook = 256, NOT the target 32).
  - `vocal_feat` ∈ {'mel'(80→d), 'envelope'(2→d)} → `nn.Linear` projection.
  - `use_x_cond` now also true when `vocal_xcond` or `vocal_feat` set.
- `_add_vocal_to_xcond(binfo, otz, vocal_feat)`: adds the vocal signal to `x_cond`
  (continuous feature takes precedence over token embed).
- `forward` / `sample` / `primed_sample`: accept and thread `vocal_feat`.

### dataset.py (BeatInfoPairedDataset)
- Reads `hps.vocal_feat`. When set, loads the `others` (vocal) mel, mean-pools
  4096→1024 frames (token rate), builds either `mel` (T,80) or `envelope` (T,2 =
  energy + positive onset flux), and appends it to the returned tuple.

### train_lm_ablation.py
- Solver.run: extracts `vocal_feat = data[4]` (if present) and passes it to the model.

### model/autoregressive.py
- Sampling: top_k and top_p applied sequentially (jukebox `filter_logits` asserts only
  one at a time). Pre-existing fix, unrelated to conditioning.

### sampling_explorer.ipynb
- Cell 1: EXP_IDX=30, CKPT_PATH='ckpt/exp30_ce_best.pkl'.
- Cell 3: extracts `vocal_feat` from dataset and passes to `primed_sample` (required —
  without it the mel model gets no conditioning). Default decode params cleaned
  (top_k=0, energy_gate=0, rep_penalty=1.0).

## Suggested next levers (not yet done)
- More data / augmentation (1486 train chunks is small).
- Richer/normalized vocal mel features; try concatenating phase + mel explicitly.
- Revisit decode params per sample: temp ~0.9, top_p ~0.95, gates off.
