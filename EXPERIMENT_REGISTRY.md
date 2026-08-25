# AttnScaf Experiment Registry

This registry defines canonical names for the headline SiT-XL/2 experiments.
Names encode scientific conditions, not the final checkpoint budget. Legacy
ablations retain their existing names until they are promoted into a paper
table.

## Naming Scheme

```text
{model}-attnscaf-{encoder}-s{scaffold_layer}-t{scaffold_duration}
-cons-{target}{weight}-smooth-{mode}
-repa-{weight}-norm-{mode}-rproj-{type}
```

- `cons-attn2`: attention-output consistency with coefficient 2.
- `cons-kv2`: direct K/V consistency with coefficient 2.
- `smooth-attn`: blend scaffolded and native attention outputs.
- `smooth-kv`: blend external and native K/V tensors.
- `repa-none`: no representation-alignment loss; `norm` and `rproj` are omitted.
- `repa{depth}-{weight}` records the REPA encoder depth and loss weight.
- `norm-none`: plain cosine REPA. In the implementation this is
  `projection-loss-type: cosine_repa`.
- `norm-zs0p6`: spatial z-score with alpha 0.6 followed by cosine alignment.
- `rproj-mlp` and `rproj-conv`: REPA projector type. The default AttnScaf K/V
  projector remains a bias-free linear map and is not repeated in the name.

## Headline Conditions

| Canonical experiment name | Legacy experiment name | Known FID | Status |
|---|---|---|---|
| `sit-xl2-attnscaf-dinob12-s8-t30k-cons-attn2-smooth-attn-repa-none` | `attnscaf-consistency-no-repa-smooth5k-layer8-100k` | 17.22 / 10.01 / 7.82 at 100K / 200K / 400K | Existing standalone headline run. |
| `sit-xl2-attnscaf-dinob12-s4-t30k-cons-attn2-smooth-attn-repa8-1-norm-zs0p6-rproj-mlp` | `attnscaf-consistency-smooth5k-repa8-scaffold4-100k` | 11.72 / 7.82 / 6.21 at 100K / 200K / 400K | Existing z-score REPA run. |
| `sit-xl2-attnscaf-dinob12-s4-t30k-cons-attn2-smooth-attn-repa8-0p5-norm-zs0p6-rproj-mlp` | `attnscaf-consistency-smooth5k-repa0.5-repa8-scaffold4-100k` | 11.20 at 100K | 11.20 association confirmed; later checkpoints still require result-file verification. |
| `sit-xl2-attnscaf-dinob12-s4-t30k-cons-attn2-smooth-attn-repa8-1-norm-none-rproj-mlp` | `attnscaf-consistency-smooth5k-plain-repa8-scaffold4-400k` | 12.00 at 100K; 6.82 at 400K | Existing plain REPA run; 200K result not recorded locally. |
| `sit-xl2-attnscaf-dinob12-s4-t30k-cons-kv2-smooth-attn-repa8-0p5-norm-none-rproj-mlp` | `attnscaf-kvmse-outputblend5k-repa8-scaffold4-100k` | 12.73 at 100K | Existing K/V-consistency control. |
| `sit-xl2-attnscaf-dinob12-s4-t30k-cons-attn2-smooth-attn-repa8-0p5-norm-none-rproj-mlp` | None | None | Matched control config created; not yet trained. |

## Provenance Rule

Before a number is moved into a paper table, record the checkpoint step, saved
checkpoint arguments, sample NPZ name, VAE decoder, sampler, number of sampling
steps, CFG, seed, GPU count, per-GPU batch size, and reference-statistics file.
Renaming an experiment directory does not rename historical sample artifacts or
external tracking runs.
