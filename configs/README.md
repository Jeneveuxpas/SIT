# Current Configs

Active revised-paper experiments for the attention-routing initialization story.

Hypothesis: DiT training is bottlenecked by poor attention routing at initialization.
Stage 1 injects pretrained encoder K/V as an attention geometry warmup.

- `sit-xl-routing-stage1-irepa.yaml`: Stage1 warmup + iREPA, no Stage2 distillation.
- `sit-xl-routing-stage1.yaml`: Stage1 warmup only.
- `sit-xl-routing-stage1-stage2-irepa.yaml`: Stage1 warmup + Stage2 distillation + iREPA.
- `sit-xl-routing-stage1-stage2.yaml`: Stage1 warmup + Stage2 distillation.

All four configs use `stage1-steps: 30000` and `max-train-steps: 100000` for a fixed training budget. Historical rejected-paper and ablation configs are kept in `../configs_legacy/`.
