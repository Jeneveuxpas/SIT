# Current Configs

Active revised-paper experiments for the AttnScaf attention-routing initialization story.

Hypothesis: DiT training is bottlenecked by poor attention routing at initialization.
Stage 1 injects pretrained encoder K/V as an attention geometry warmup, and Stage 2 distills the native SiT attention output back to the encoder-guided target.

Core routing ablations:

- `sit-xl-routing-stage1.yaml`: Stage1 warmup only.
- `sit-xl-routing-stage1-irepa.yaml`: Stage1 warmup + iREPA.
- `sit-xl-routing-stage1-stage2.yaml`: Stage1 warmup + Stage2 distillation.
- `sit-xl-routing-stage1-stage2-irepa.yaml`: Stage1 warmup + Stage2 distillation + iREPA.

KV-norm ablations on the Stage1 + Stage2 + iREPA main setting. The original `none` setting has already been run as the default baseline:

- `sit-xl-attnscaf-s1s2-irepa-kvnorm-layernorm.yaml`
- `sit-xl-attnscaf-s1s2-irepa-kvnorm-rmsnorm.yaml`
- `sit-xl-attnscaf-s1s2-irepa-kvnorm-k-rms-v-layer.yaml`
- `sit-xl-attnscaf-s1s2-irepa-kvnorm-k-layer-v-rms.yaml`

All current configs use `stage1-steps: 30000` and `max-train-steps: 100000`. iREPA configs use `proj-coeff: "1.0"`. Historical rejected-paper and ablation configs are kept in `../configs_legacy/`.
