# `configs/` — experiment configs

One YAML per trained model. Each fully specifies data, model, loss, and training, so
`python3 src/train.py --config <this yaml>` reproduces that run. Output goes to the
config's `experiment.output_dir` / `experiment.name`.

## `main/` — the four published models
| Config | Model | Domain |
|--------|-------|--------|
| `full_domain.yaml` | full-domain production model | 800×800, 12 levels, high_weight=0, non-negativity on |
| `spatial_transfer1.yaml` | spatial transfer, interior tile (T4) | trains on surrounding tiles, tests on held-out `[200:400, 200:400]` |
| `spatial_transfer2.yaml` | spatial transfer, central-London tile | held-out `[0:200, 400:600]` (extreme high-emission block) |
| `future_transfer.yaml` | future transfer (temporal extrapolation) | train Jul+Aug, test held-out Sep (report Sep 4–6: RMSE 1.068 / r 0.766) |

All four share the final recipe: z0–11, `high_weight=0`, `lambda_nonneg=0.1`, all loss terms, 300 epochs.

## `input_importance_800/` — input leave-one-out (notebook 02)
Ten configs, each zeroing one physical variable's *t* and *t−1* channels (`data.zero_channels`)
and retraining the full-domain model. `ii800_t_only` drops the whole t−1 block; `ii800_no_height`
blanks the height encoding (`model.zero_height`). Baseline = `main/full_domain`.

## `loss_ablation_800/` — loss leave-one-out (notebook 03)
Four configs, each removing one loss term from the full-domain recipe and retraining
(`la800_no_{anom,grad,pattern,spectral}`). Baseline = `main/full_domain`;
the `no_nonneg` / `high_weight=8` comparisons reuse existing runs.

## `transfer_ratio/` — transfer vs training coverage (notebook 04)
Six configs training each held-out tile on a growing neighbourhood of tiles
(`data.train_include_tiles`): `transfer{1,2}_{1,3,8}v1` (1/3/8 training tiles).
The `15v1` point reuses the published `spatial_transfer{1,2}`. Layouts: 1v1 = one
left-neighbour tile; 3v1 = 2×2 with the test tile bottom-right; 8v1 = 3×3 with the
test tile bottom-centre.
