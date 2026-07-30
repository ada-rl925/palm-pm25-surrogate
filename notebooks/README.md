# `notebooks/`

Analysis and results notebooks. Executed top-to-bottom; they read model weights from
`../../outputs/` and metric summaries from the `*.json` files there.

## `results/` — per-model outcome notebooks
Truth vs prediction, relative-error maps, and per-layer metrics for each published model.
Three example snapshots (idx49 calm night, idx59 noon, idx39 hardest afternoon), each as a
per-level `truth | prediction | %-difference` composite.

| Notebook | Model | z0–11 pooled |
|----------|-------|--------------|
| `outcome_full_domain.ipynb` | `full_domain` | RMSE 0.755 / r 0.887 / R² 0.940 |
| `outcome_spatial_transfer1.ipynb` | `spatial_transfer1` (T4 tile) | RMSE 0.546 / r 0.511 |
| `outcome_spatial_transfer2.ipynb` | `spatial_transfer2` (centre tile) | RMSE 2.628 / r 0.713 |

## `experiments/` — analysis studies
| Notebook | What |
|----------|------|
| `01_eda_preprocessing.ipynb` | Study-area map, domain morphology, input distributions & normalisation choices, the t/t−1 history, target statistics, building mask, channel layout. |
| `02_input_importance.ipynb` | Leave-one-out retraining of each input (full domain): overall + per-layer RMSE/r, plus a model-agnostic Random Forest cross-check. |
| `03_loss_ablation.ipynb` | Leave-one-out retraining of each loss term (full domain): overall + per-layer RMSE/r/σ. |
| `04_transfer_scaling.ipynb` | Transfer accuracy vs training coverage: both held-out tiles predicted by models trained on 1/3/8/15 neighbouring tiles. |

`experiments/figs/` holds committed figures embedded in the notebooks and used in the paper.

Ablation metrics are stored as small JSON under `outputs/ablation_{input,loss}/`.
