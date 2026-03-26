# wosac_eval

Minimal offline WOSAC evaluation project extracted from TrajTok.

## What It Does

This project is focused on two tasks only:

1. Build GT pickle files for Waymo validation scenarios.
2. Evaluate rollout pickle files offline with the fast WOSAC metric implementation.

## Expected Rollout Format

Each rollout file must be named as `<scenario_id>.pkl` and contain:

- `agent_id`: shape `[n_agents]`
- `simulated_states`: shape `[n_rollout, n_agents, n_step, 4]`

The last dimension of `simulated_states` must be `(x, y, z, yaw)` in global coordinates.

## Install

```bash
conda create -y -n wosac_eval python=3.11.9
conda activate wosac_eval
pip install -r requirements.txt
pip install --no-deps waymo-open-dataset-tf-2-12-0==1.6.4
```

## Prepare GT Pickles

```bash
python prepare_gt.py /path/to/waymo/scenario/validation --output-dir data/waymo_processed/validation_gt
```

This writes one GT pickle per scenario into the output directory.

## Evaluate Rollouts Offline

```bash
python wosac_eval.py /path/to/rollout_dir --gt-dir data/waymo_processed/validation_gt --version 2025
```

This writes a JSON report to:

```bash
/path/to/rollout_dir/wosac_metrics_report.json
```

You can also point it at an external rollout directory, for example:

```bash
python wosac_eval.py /home/hcis-s26/Yuhsiang/catk/vis/catk_sketch_1000 --gt-dir data/waymo_processed/validation_gt --version 2025
```

## Output Report

The JSON report includes:

- aggregate metrics
- number of matched files
- failed files with error messages
- files missing GT
- files missing rollout
