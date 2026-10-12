# `outputs/` — run artifacts

Per-run outputs written by `src/train.py`: `config.yaml`, `metrics.csv`, `test_results.json`
and `checkpoints/best.pt`. The directory name equals the config's `experiment.name`, so each
model is self-describing. All released models are mask models (`in_ch=19`, ~1.6 M params).

## Published models (weights included)
| Directory | Model | Test (pooled over valid cells) |
|-----------|-------|--------------------------------|
| `full_domain_mask/` | full-domain, temporal generalisation | RMSE 0.729 / r 0.889 |
| `future_transfer_mask/` | future transfer, train 2019 Jul–Sep → test 2020 | RMSE 1.32 / r 0.83 (4–6 Sep 2020) |
| `spatial_transfer1_mask/` | spatial transfer, interior tile (never seen) | RMSE 0.554 / r 0.496 |
| `spatial_transfer2_mask/` | spatial transfer, central-London tile (never seen) | RMSE 2.700 / r 0.703 |

## Training-data scaling (notebook 05)
Evaluated on the held-out January 2019 winter episode; baseline = each model's own
training-set climatology.
| Directory | Training data | Surrogate RMSE / r | Baseline RMSE / r |
|-----------|---------------|--------------------|-------------------|
| `data_scaling/ft_96/`  | 96 h (Jul)      | 7.25 / 0.53 | 2.55 / 0.25 |
| `future_transfer_mask/`| 336 h (Jul–Sep) | 1.02 / 0.83 | 2.39 / 0.42 |
| `data_scaling/ft_600/` | 600 h (Jul–Oct) | 0.72 / 0.84 | 2.64 / 0.52 |

`data_scaling/climatology.npz` holds the three training-set climatology fields used for the baseline.

## Ablations (best.pt kept; results also summarised as JSON)
| Directory | Study | Notebook | Summary JSON |
|-----------|-------|----------|--------------|
| `input_ablation/` | input leave-one-out retraining (10 runs `ii800_*`) | `02_input_ablation` | `input_importance_layer_mask.json` |
| `loss_ablation/`  | loss leave-one-out retraining (5 runs `la800_*`)   | `03_loss_ablation`  | `loss_ablation_layer_mask.json` |

The `*_layer_mask.json` files hold pooled **and per-layer** RMSE/r for every variant (plus the
baseline); the notebooks read only these, so they render without the ablation checkpoints.

## Other files
- `heldout_eval_future_transfer_mask.json` — per-episode held-out metrics for the future-transfer model (read by `results/outcome_future_transfer_mask`).
- `climatology_future_transfer_mask.npy` — the future-transfer model's training-set climatology (baseline in the same notebook).
