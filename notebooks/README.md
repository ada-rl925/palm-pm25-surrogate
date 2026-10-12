# `notebooks/`

Analysis and results notebooks. Executed top-to-bottom; they read model weights from
`../../outputs/` (and metric summaries from the `*.json` files there).

## `results/` — per-model outcome notebooks
Truth vs prediction, relative-error maps and per-layer metrics for each published model, with
per-level `PALM | prediction | difference` composites for selected frames.

| Notebook | Model | Pooled test (z0–11) |
|----------|-------|---------------------|
| `outcome_full_domain_mask.ipynb` | `full_domain_mask` (temporal generalisation) | RMSE 0.729 / r 0.889 |
| `outcome_future_transfer_mask.ipynb` | `future_transfer_mask` (train 2019 → test 2020) | RMSE 1.32 / r 0.83 (4–6 Sep 2020) |
| `outcome_spatial_transfer1_mask.ipynb` | `spatial_transfer1_mask` (interior tile) | RMSE 0.554 / r 0.496 |
| `outcome_spatial_transfer2_mask.ipynb` | `spatial_transfer2_mask` (centre tile) | RMSE 2.700 / r 0.703 |

## `experiments/` — analysis studies
| Notebook | What |
|----------|------|
| `01_eda_preprocessing.ipynb` | Study-area map, domain morphology, input distributions & normalisation, the t/t−1 history, target statistics, building mask, channel layout. |
| `02_input_ablation.ipynb` | Leave-one-out **retraining** of each input (full domain): overall + per-layer RMSE/r. |
| `03_loss_ablation.ipynb` | Leave-one-out **retraining** of each loss term (full domain): overall + per-layer RMSE/r. |
| `04_physics_consistency.ipynb` | Inference-time perturbations of the full-domain model: emission scaling, wind-speed (dilution) scaling, wind-direction rotation. |
| `05_data_scaling.ipynb` | Training-data scaling (96 / 336 / 600 h) evaluated on the held-out January 2019 winter episode, vs a climatological baseline. |

`experiments/figs/` and `results/figs/` hold the committed figures embedded in the notebooks
and used in the paper.
