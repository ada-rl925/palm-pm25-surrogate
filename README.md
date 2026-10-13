# 3D Deep Learning Surrogate for Building-Resolving Urban PM2.5

A physics-guided 3D U-Net surrogate that emulates building-resolving PALM large-eddy simulation (LES) of urban PM2.5 from high-resolution emissions, urban geometry and large-scale meteorological forcing.

The model, `UNet3DFiLM`, emulates PALM LES of PM2.5 over central London: an 8 km × 8 km domain over Camden (British National Grid E 523850–531850, N 180050–188050), resolved on an 800 × 800 horizontal grid at 10 m horizontal resolution and 12 vertical levels from 15 m to 195 m. The surrogate maps hourly physical drivers to three-dimensional PM2.5 fields, producing a full 3D field in about 255 ms on a single NVIDIA RTX A5000 GPU, approximately 3800× faster in wall-clock time than the PALM simulations it emulates.

All released models take the binary occupancy mask (1 = outdoor atmospheric cell, 0 = terrain or building) as an explicit input.

## Highlights

- Physics-guided 3D U-Net surrogate for building-resolving PM2.5 at 10 m horizontal resolution
- ~3800× faster than PALM LES in wall-clock time (one-time training cost of ~19 GPU hours)
- Temporal generalisation to held-out hours of the 2019 episodes
- Future period transferability across years (train 2019 → test September 2020)
- Spatial transferability to unseen tiles within the domain
- Input-feature and loss-component ablations, physical-consistency tests and training data scaling

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

The PALM dataset is not distributed in this repository. `src/dataloader.py` expects it under `data/` at the repository root (see **Data** below).

## Released models (weights included)

Every config above has its trained weights under `outputs/<name>/checkpoints/best.pt` (config name = output directory, so each model is self-describing).

| Model | Config / weights | Test setting | RMSE / r |
|---|---|---|---|
| **Temporal generalisation** | `full_domain_mask` | held-out days (20 July, 21 August, 6 September 2019), full 800×800 domain, 12 levels | **0.729 / 0.889** |
| **Future period transfer** | `future_transfer_mask` | trained on the three 2019 episodes → tested on 4–5 September 2020 | **1.315 / 0.834** |
| **Spatial transfer, tile 1** | `spatial_transfer1_mask` | held-out tile near the domain centre `[200:400, 200:400]`, 15 source tiles | **0.554 / 0.496** |
| **Spatial transfer, tile 2** | `spatial_transfer2_mask` | held-out tile in the lower half of the domain `[0:200, 400:600]`, 15 source tiles | **2.700 / 0.703** |

RMSE is pooled over valid outdoor cells in µg/m³; r is the mean per-level pattern correlation. All metrics quantify fidelity to the PALM reference, not accuracy against observations.

### Training data scaling

Surrogates trained on increasing amounts of 2019 data are evaluated on an independent 48 h winter episode starting on 8 January 2019, which lies outside the meteorological conditions of all training sets (`05_data_scaling`). The baseline is each model's own training-set climatology.

| Training data | Weights | Surrogate RMSE / r | Baseline RMSE / r |
|---|---|---|---|
| 96 h (Jul)      | `data_scaling/ft_96`      | 7.25 / 0.53 | 2.55 / 0.25 |
| 336 h (Jul–Sep) | `future_transfer_mask`    | 1.02 / 0.83 | 2.39 / 0.42 |
| 600 h (Jul–Oct) | `data_scaling/ft_600`     | 0.72 / 0.84 | 2.64 / 0.52 |

The 336 h point is the future period transfer model itself.

## Model

- **Backbone**: 3D U-Net with anisotropic convolutions (horizontal 1×3×3 followed by vertical 3×1×1) and horizontal-only pooling, so that the vertical resolution is preserved. The full-domain models use `n_down=4` (50×50 bottleneck); the 200×200 spatial-transfer tiles use `n_down=3`.
- **Spatial input**: the PM2.5 emission fields at t and t−1, the model-level height field and the binary occupancy mask are supplied directly to the convolutional encoder as a four-channel tensor of shape (4, 12, 800, 800). The 2D surface emission is placed at the lowest atmospheric model level above the local terrain and building surface.
- **FiLM conditioning**: the ERA5 wind, potential temperature and specific humidity profiles, the CAMS background PM2.5 profile, surface pressure and the hour-of-day encoding, at t and t−1, modulate the feature maps through per-channel, per-level scale and shift parameters.
- **Bottleneck self-attention**: per-level spatial self-attention captures long-range spatial dependencies.
- **Composite loss**: MSE data term + 0.1 × non-negativity + spatial anomaly + gradient + pattern correlation + spectral consistency, all evaluated over valid outdoor cells only.

The surrogate is physics-guided in the sense that physical knowledge shapes its input representation, network design, loss and restriction to the outdoor domain. It does not explicitly solve or enforce the governing flow and transport equations.

## Ablation studies

- **Input importance** (`02_input_ablation`): boundary PM2.5 is by far the most important input and becomes increasingly dominant with height, while the surface emission is the principal near-surface input. The horizontal winds, especially the meridional component, mainly control the spatial structure at upper levels, and the preceding-hour inputs become increasingly important aloft.
- **Loss components** (`03_loss_ablation`): the pattern-correlation term makes the largest contribution to both accuracy and spatial correlation. The spectral, anomaly and non-negativity terms give comparable accuracy gains, while the gradient term contributes least to accuracy. The effects on correlation are most pronounced at the upper levels.

Both analyses render directly from the JSON summaries in `outputs/{input,loss}_ablation/`, so the per-run ablation checkpoints are not required to reproduce the figures.

## Physical consistency

`04_physics_consistency` perturbs the trained full-domain model at inference. Scaling the emission field increases the near-source concentration monotonically, and removing it returns the concentration to the background level. Scaling the wind speed from 0.5 to 3 times its original value reduces the 99th percentile concentration by about 42%, consistent with stronger dilution. Rotating the wind vector re-orients the predicted plume with the imposed direction. These tests are qualitative sensitivity checks, since some perturbations extend beyond the forcing range represented in the training data.

## Data

The code and trained models are self-contained, but the PALM dataset itself is not distributed with this repository because of its size. `src/dataloader.py` expects the compact dataset under `data/` at the repository root (override the location with the `PALM_DATA_DIR` environment variable). Each episode directory holds `INPUT/` (chemistry and dynamic drivers) and the `av_xy` output (PM2.5), together with `emission_z.npy` and `topo.npy`. It is a lossless subset of the raw PALM output, containing only what the surrogate reads.

The PALM simulations are driven by ERA5 meteorological forcing and CAMS background PM2.5 profiles; surface emissions derived from the NAEI and CAMS-REG-TEMPO inventories; and a static driver built from OS Terrain 50, Environment Agency 1 m National LiDAR, Ordnance Survey topography and building height data, and ESA WorldCover. Some of these input datasets are openly available, while others are proprietary and subject to their own licences.

**The dataset is available from the authors on request.**

## Reproduce

```bash
pip install -r requirements.txt
# obtain the dataset from the authors and place it under ./data, then:
python3 src/train.py --config configs/main/full_domain_mask.yaml
# other models: configs/main/future_transfer_mask.yaml, configs/main/spatial_transfer1_mask.yaml,
#               configs/main/spatial_transfer2_mask.yaml, configs/data_scaling/ft_96.yaml,
#               configs/input_ablation/*.yaml, configs/loss_ablation/*.yaml
# (to read a full raw PALM tree instead of ./data, set PALM_DATA_DIR)
```

## AI Usage Declaration

Claude (Anthropic) was used during this project to assist with code development and debugging.
