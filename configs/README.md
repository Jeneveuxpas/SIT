# Current Configs

Active revised-paper experiments for the AttnScaf attention-routing initialization story.

Hypothesis: DiT training is bottlenecked by poor attention routing at initialization.
Stage 1 injects pretrained encoder K/V as an attention geometry warmup. After the warmup, training continues with native SiT attention and iREPA; Stage 2 distillation is kept only as an ablation because its marginal gain is small.

Core routing ablations:

- `sit-xl-routing-stage1.yaml`: Stage1 warmup only.
- `sit-xl-routing-stage1-irepa.yaml`: Stage1 warmup + iREPA.
- `sit-xl-routing-stage1-stage2.yaml`: Stage1 warmup + Stage2 distillation ablation.
- `sit-xl-routing-stage1-stage2-irepa.yaml`: Stage1 warmup + Stage2 distillation + iREPA ablation.

KV-norm ablations on the Stage1 + iREPA main setting. Stage 2 distillation is disabled in these configs (`distill-coeff: 0.0`, `kv-stop-step: 30000`). The original `none` setting has already been run as the default baseline:

- `sit-xl-attnscaf-stage1-irepa-kvnorm-layernorm.yaml`
- `sit-xl-attnscaf-stage1-irepa-kvnorm-rmsnorm.yaml`
- `sit-xl-attnscaf-stage1-irepa-kvnorm-k-rms-v-layer.yaml`
- `sit-xl-attnscaf-stage1-irepa-kvnorm-k-layer-v-rms.yaml`

All current configs use `stage1-steps: 30000` and `max-train-steps: 100000`. iREPA configs use `proj-coeff: "1.0"`. Historical rejected-paper and ablation configs are kept in `../configs_legacy/`.
