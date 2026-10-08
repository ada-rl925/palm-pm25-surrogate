# `outputs/` — run artifacts

Per-run outputs written by `src/train.py`: `config.yaml`, `metrics.csv`, `test_results.json`,
and `checkpoints/`. The **best checkpoint (`best.pt`) of every run is kept**; `last.pt` and
any other weights are git-ignored (see `.gitignore`).

## Published models (weights included)
| Directory | Model | Test (z0–11, pooled) |
|-----------|-------|----------------------|
| `full_domain/` | full-domain production model (~1.6 M params) | RMSE 0.755 / r 0.887 / R² 0.940 |
| `spatial_transfer1/` | transfer, interior tile T4 (never seen) | RMSE 0.546 / r 0.511 |
| `spatial_transfer2/` | transfer, central-London tile (never seen) | RMSE 2.628 / r 0.713 |
| `future_transfer/` | temporal extrapolation, train Jul+Aug / test held-out Sep | RMSE 1.068 / r 0.766 (Sep 4–6) |

## Ablations (best.pt kept; results also summarised as JSON)
| Directory | Study | Notebook | Summary JSON |
|-----------|-------|----------|--------------|
| `ablation_input/` | input leave-one-out retraining (10 runs `ii800_*`) | `02_input_importance` | `input_importance_layer.json`, `rf_importance.json` |
| `ablation_loss/` | loss leave-one-out retraining (4 runs `la800_*`) | `03_loss_ablation` | `loss_ablation_layer.json` |
| `transfer_ratio/` | transfer vs training coverage (6 runs `transfer{1,2}_{1,3,8}v1`) | `04_transfer_scaling` | `transfer_ratio_metrics.json` |

The `*_layer.json` files hold pooled **and per-layer** RMSE/r/σ for every variant (plus the
baseline); the notebooks read only these, so they render without the ablation checkpoints.

**Key ablation findings** (full domain): amplitude is set by the CAMS boundary inflow
(largest ΔRMSE), spatial structure by the wind (largest Δr, v > u); both the t−1 history and
the pattern/spectral loss terms matter mainly at the **upper levels (135–195 m)**.

## Other
- `figs/` — committed figures used across notebooks and the paper.
- `train_logs/` — stdout logs of the ablation runs.
- `_geo_cache/` — cached OSM geojson from `src/draw_study_area_map.py` (git-ignored).
