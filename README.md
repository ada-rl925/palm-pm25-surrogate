# 3D Machine Learning Surrogate for Urban PM2.5 Prediction

A physics-informed 3D U-Net surrogate for building-resolving urban PM2.5 prediction from high-resolution emissions, urban morphology and meteorological forcing.

The proposed `UNet3DFiLM` emulates PALM large-eddy simulation (LES) of PM2.5 concentration over central London: an 8 km × 8 km domain over Camden (British National Grid E 523850–531850, N 180050–188050), resolved on an 800 × 800 horizontal grid at 10 m resolution and 12 vertical levels from 15 m to 195 m. The surrogate maps hourly physical drivers to three-dimensional PM2.5 fields while running approximately 3800× faster than the PALM simulations it emulates.

All released models take the valid/building **mask** as an explicit input (`in_ch=19`, `spatial_ch=3`).

## Highlights

- Physics-informed 3D U-Net surrogate for building-resolving PM2.5 prediction
- ~3800× faster than PALM LES
- Temporal generalisation to unseen meteorological conditions
- Future transfer across years (train 2019 → test 2020)
- Spatial transfer to neighbouring urban regions
- Input-feature and loss-function ablations; physical-consistency checks; training-data scaling

## Repository layout

```text
src/            model.py (UNet3DFiLM + PhysicsLoss), dataloader.py, train.py        [src/README.md]
                draw_study_area_map.py (standalone helper for the study-area figure)

configs/        one YAML per experiment                                             [configs/README.md]
  main/         full_domain_mask, future_transfer_mask, spatial_transfer1_mask,
                spatial_transfer2_mask  (the four published models)
  data_scaling/                ft_96 (96 h) and ft_600 (600 h) scaling points
  input_ablation/              leave-one-out input-feature ablation (10 runs)
  loss_ablation/               leave-one-out loss-component ablation (5 runs)

notebooks/                                                                          [notebooks/README.md]
  experiments/  01_eda_preprocessing, 02_input_ablation, 03_loss_ablation,
                04_physics_consistency, 05_data_scaling
  results/      outcome_full_domain_mask, outcome_future_transfer_mask,
                outcome_spatial_transfer1_mask, outcome_spatial_transfer2_mask
                (prediction / error maps, per-level metrics, case figures in figs/)

outputs/        per-model config, metrics.csv, test_results.json, checkpoints       [outputs/README.md]
```

The PALM **dataset is not distributed in this repository** — `src/dataloader.py` expects it
under `data/` at the repo root (contact the author to obtain it; see **Data** below).

## Released models (weights included)

Every config above has its trained weights under `outputs/<name>/checkpoints/best.pt`
(config name = output directory, so each model is self-describing).

| Model | config / weights | Test setting | RMSE / r |
|---|---|---|---|
| **Temporal generalisation** | `full_domain_mask` | held-out hours, full 800×800, 12 levels | **0.729 / 0.889** |
| **Future transfer** | `future_transfer_mask` | train 2019 Jul–Sep → test Sep **2020** | **1.32 / 0.83** (4–6 Sep; 7 Sep is an anomalous day) |
| **Spatial transfer 1** | `spatial_transfer1_mask` | held-out interior tile `[200:400, 200:400]` | **0.554 / 0.496** |
| **Spatial transfer 2** | `spatial_transfer2_mask` | held-out centre tile `[0:200, 400:600]` | **2.700 / 0.703** |

RMSE is pooled over valid cells in µg/m³; r is the mean per-level pattern correlation.

### Training-data scaling

Surrogates trained on increasing amounts of 2019 data, evaluated on the held-out
**January 2019 winter episode** (`05_data_scaling`). Baseline = each model's own
training-set climatology.

| Training data | weights | Surrogate RMSE / r | Baseline RMSE / r |
|---|---|---|---|
| 96 h (Jul)      | `data_scaling/ft_96`      | 7.25 / 0.53 | 2.55 / 0.25 |
| 336 h (Jul–Sep) | `future_transfer_mask`    | 1.02 / 0.83 | 2.39 / 0.42 |
| 600 h (Jul–Oct) | `data_scaling/ft_600`     | 0.72 / 0.84 | 2.64 / 0.52 |

The 336 h point is the future-transfer model itself.

## Model

- **Backbone**: 3D U-Net with anisotropic convolutions and horizontal-only pooling (vertical resolution preserved); full-domain model uses `n_down=4` (50×50 bottleneck), the 200×200 spatial-transfer tiles use `n_down=3`.
- **FiLM conditioning**: ERA5 winds / θ / qv / PM2.5 profiles + surface pressure + hour encoding modulate the feature maps after each convolution block through per-channel, per-level γ and β.
- **Bottleneck self-attention**: per-level spatial attention captures long-range spatial structure.
- **Input**: `in_ch=19` = 9 physical variables × {t, t−1} (18) + the valid/building mask; `spatial_ch=3` = building height, terrain elevation, mask.
- **Composite loss**: weighted MSE (peak up-weighting) + non-negativity + spatial anomaly + gradient + pattern-correlation + spectral consistency.

## Ablation studies

- **Input importance** (`02_input_ablation`): boundary PM2.5 and surface emissions are the dominant inputs overall. Horizontal winds mainly determine the spatial structure, particularly at higher levels, while the preceding-hour inputs become increasingly important aloft.
- **Loss components** (`03_loss_ablation`): anomaly and gradient terms mainly improve accuracy and local sharpness, whereas the pattern-correlation and spectral terms preserve spatial structure, particularly at higher levels; the non-negativity term is a physical constraint with little effect on the reported metrics.

Both render directly from the JSON summaries in `outputs/{input,loss}_ablation/`, so the per-run ablation checkpoints are not required to reproduce the figures.

## Physical consistency

`04_physics_consistency` perturbs the trained full-domain model at inference: scaling emissions, scaling wind speed (dilution) and rotating wind direction. The surrogate's response follows the expected physical trends (monotonic emission response, dilution under stronger winds, plume re-orientation with wind direction).

## Data

The code and trained models here are self-contained, but the **PALM dataset itself is not
distributed with this repository.** `src/dataloader.py` expects the compact dataset under
`data/` at the repo root (override the location with the `PALM_DATA_DIR` environment variable).
Each episode directory holds `INPUT/` (chemistry + dynamic drivers) and the `av_xy` output
(PM2.5), plus `emission_z.npy` and `topo.npy`; it is a lossless subset of the raw PALM output
(only what the surrogate reads), built from PALM output and openly licensed ERA5, CAMS, NAEI
and CAMS-REG-TEMPO.

**The dataset is available on request — please contact the author.**

## Reproduce

```bash
pip install -r requirements.txt
# obtain the dataset from the author and place it under ./data, then:
python3 src/train.py --config configs/main/full_domain_mask.yaml
# other models: configs/main/future_transfer_mask.yaml, configs/main/spatial_transfer1_mask.yaml,
#               configs/data_scaling/ft_96.yaml, configs/input_ablation/*.yaml, ...
# (to read a full raw PALM tree instead of ./data, set PALM_DATA_DIR)
```

## AI Usage Declaration

Claude (Anthropic) was used during this project to assist with code development and debugging.
