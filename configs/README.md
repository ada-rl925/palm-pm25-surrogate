# `configs/` — experiment configs

One YAML per trained model. Each fully specifies data, model, loss and training, so
`python3 src/train.py --config <this yaml>` reproduces that run. Output goes to the
config's `experiment.output_dir` / `experiment.name`. All released models take the
valid/building mask as an explicit input (`in_ch=19`, `spatial_ch=3`).

## `main/` — the four published models
| Config | Model | Test setting |
|--------|-------|--------------|
| `full_domain_mask.yaml` | full-domain, temporal generalisation | 800×800, 12 levels; held-out hours within 2019 |
| `future_transfer_mask.yaml` | future transfer (train 2019 Jul–Sep) | tested on Sep **2020** (also the 336 h data-scaling point) |
| `spatial_transfer1_mask.yaml` | spatial transfer, interior tile | train surrounding tiles, test held-out `[200:400, 200:400]` |
| `spatial_transfer2_mask.yaml` | spatial transfer, central-London tile | held-out `[0:200, 400:600]` (high-emission block) |

All share the final recipe: z0–11, `high_weight=0`, `lambda_nonneg=0.1`, all loss terms, 300 epochs.

## `data_scaling/` — training-data scaling (notebook 05)
Surrogates trained on increasing amounts of 2019 data, evaluated on the held-out January
2019 winter episode:
| Config | Training data |
|--------|---------------|
| `ft_96.yaml`  | 96 h (Jul) |
| (336 h point) | = `main/future_transfer_mask.yaml` |
| `ft_600.yaml` | 600 h (Jul–Oct, all 2019) |

## `input_ablation/` — input leave-one-out (notebook 02)
Ten configs, each zeroing one physical variable's *t* and *t−1* channels (`data.zero_channels`)
and retraining the full-domain model. `ii800_t_only` drops the whole t−1 block; `ii800_no_height`
blanks the height encoding. Baseline = `main/full_domain_mask`.

## `loss_ablation/` — loss leave-one-out (notebook 03)
Five configs, each removing one loss term from the full-domain recipe and retraining
(`la800_no_{anom,grad,pattern,spectral,nonneg}`). Baseline = `main/full_domain_mask`.
