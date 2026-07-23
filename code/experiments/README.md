# Phase-one experiment workflow

The experiment contract is defined in `experiment_config.py`: all methods use
the same 50 Hz environment, 7-dimensional structured action interface where
applicable, five training seeds, and 100 fixed evaluation episodes per seed.

Generate the executable job manifest:

```bash
cd code
python -m experiments.manifest --output ../results/phase1/manifest.json
```

Train the jobs in `training_jobs` from the manifest. The structured variants
use `train.py --agent_type structured`; direct joint SAC uses `joint`, and the
unstructured residual baseline uses `residual`.

Run each `evaluation_jobs` command after its checkpoint is available. Every
command writes canonical JSONL rows. Aggregate all shards and produce the
paper table:

```bash
python -m experiments.report ../results/phase1/*/*/*.jsonl \
  --output-dir ../paper/generated
```

The LaTeX paper includes `paper/generated/phase1_results.tex` only when it
exists. No result number is inferred or filled in by the code.
