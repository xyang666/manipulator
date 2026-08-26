# RL Challenge v1 multi-seed results

These are the frozen 200-scene test results used by
`PAPER_EXPERIMENT_STATUS_2026-08-26.md`. Each JSONL contains exactly one row per
scene in `results/ewalker_scenes/rl_challenge_v1/test.json`.

Files:

- `adaptive_gradient_cbf.jsonl`: deterministic strongest classical baseline;
- `v12_seed11.jsonl`, `v12_seed22.jsonl`, `v12_seed33.jsonl`: validation-selected
  RL v12 checkpoints;
- `v11_seed11.jsonl`, `v11_seed22.jsonl`, `v11_seed33.jsonl`: matched checkpoints
  trained without the CBF intervention regularizer;
- `paper_statistics.json`: coverage-checked seed summaries and paired tests.
- `intervention_ablation_statistics.json`: matched-seed v11/v12 ablation.

Regenerate the statistical report from the repository root:

```bash
cd code
.venv/bin/python -m experiments.compare_paper_results \
  --scene-json ../results/ewalker_scenes/rl_challenge_v1/test.json \
  --rl ../results/rl_challenge_v1_multiseed_2026-08-26/v12_seed11.jsonl \
       ../results/rl_challenge_v1_multiseed_2026-08-26/v12_seed22.jsonl \
       ../results/rl_challenge_v1_multiseed_2026-08-26/v12_seed33.jsonl \
  --baseline ../results/rl_challenge_v1_multiseed_2026-08-26/adaptive_gradient_cbf.jsonl \
  --output ../results/rl_challenge_v1_multiseed_2026-08-26/paper_statistics.json
```

Regenerate the matched intervention ablation:

```bash
cd code
.venv/bin/python -m experiments.compare_ablation_results \
  --scene-json ../results/ewalker_scenes/rl_challenge_v1/test.json \
  --control ../results/rl_challenge_v1_multiseed_2026-08-26/v11_seed11.jsonl \
            ../results/rl_challenge_v1_multiseed_2026-08-26/v11_seed22.jsonl \
            ../results/rl_challenge_v1_multiseed_2026-08-26/v11_seed33.jsonl \
  --treatment ../results/rl_challenge_v1_multiseed_2026-08-26/v12_seed11.jsonl \
              ../results/rl_challenge_v1_multiseed_2026-08-26/v12_seed22.jsonl \
              ../results/rl_challenge_v1_multiseed_2026-08-26/v12_seed33.jsonl \
  --output ../results/rl_challenge_v1_multiseed_2026-08-26/intervention_ablation_statistics.json
```

SHA-256 checksums:

```text
4347b5cd1a090b651e65e637fb8140cc3cbf33bad1040876f0f57c16f4d0c913  test.json
536afc4bbca1c0dc5207df5ec899278bf42bc799a86baec42c7b558b9020275c  adaptive_gradient_cbf.jsonl
cb7ec4d4969d0b8d6c2ae434294c5e6ba3aa6f2d0c502d40cced0e038a3d7c38  v12_seed11.jsonl
06c9f66d99234c8905a63ed6d4bb3fe772a7fec7a9be25e592fdd2032660102c  v12_seed22.jsonl
a6fda6abd7c344143eb07bd9b7d6579e8d67813440f6f1103df41c7799cd09b7  v12_seed33.jsonl
b7036aef0d527a388776f932f5682caf2bc358f58f56ba22686c532faedd655a  paper_statistics.json
657b38cd274447b02982d8fbefb2e5ae5769bcf4f852cb94e4b35b4bb8546244  v11_seed11.jsonl
85f0d64e350c7a09bc22244ae5318875f505b55747dcaf174d29f3cdbf39c5cc  v11_seed22.jsonl
e8f8ba315823dc2da4256149709253c9ac01b88ac0d95c448192300f8b8ae7bc  v11_seed33.jsonl
416de82c4c9bf362994b89730b78df71ee6d3157bda1b1230b5d730059263624  intervention_ablation_statistics.json
```
