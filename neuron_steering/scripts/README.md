# Neuron Steering Scripts

The bash runners in `neuron_steering` are grouped by workflow:

- `extraction/`: LAPE neuron extraction and analysis jobs.
- `generation/`: neuron-steered generation and ablation runs.
- `coverage/`: LAPE/vector coverage and residual-projection measurements.
- `coverage/residual_projection_random_exclude_jobs/`: per-model, per-dialect residual-projection jobs.

Run scripts from the repository root, for example:

```bash
bash neuron_steering/scripts/generation/run_all_dialects_explicit_prompt_inference.sh --dry-run
bash neuron_steering/scripts/coverage/run_all_dialects_lape_residual_projection_allam_fanar_instruct.sh
```

Each script derives the repository and `neuron_steering` paths from its own location, so it does not depend on the current working directory.
