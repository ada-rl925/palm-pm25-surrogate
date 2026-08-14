# 3-D Machine Learning Surrogate for Urban PM2.5 Prediction

A physics-informed 3D U-Net surrogate for building-resolving urban PM2.5 prediction from high-resolution emissions, urban morphology and meteorological forcing.

The proposed `UNet3DFiLM` emulates PALM large-eddy simulation (LES) of PM2.5 concentration over central London: an 8 km × 8 km domain over Camden (British National Grid E 523850–531850, N 180050–188050), resolved on an 800 × 800 horizontal grid at 10 m resolution and 12 vertical levels from 15 m to 195 m. The surrogate maps hourly physical drivers to three-dimensional PM2.5 fields while running approximately 3800× faster than the PALM simulations it emulates.

## Highlights

- Physics-informed 3D U-Net surrogate for building-resolving PM2.5 prediction
- 3800× faster than PALM LES
- Temporal generalisation to unseen meteorological conditions
- Spatial transfer to neighbouring urban regions
- Feature and loss-function ablation studies

## Repository layout

```text
src/            model.py (UNet3DFiLM + PhysicsLoss), dataloader.py,
                train.py, utils.py                                        [src/README.md]
                draw_study_area_map.py (standalone helper for the study-area figure)

configs/        one YAML configuration per experiment                     [configs/README.md]
  main/                 final models: full_domain, spatial_transfer1, spatial_transfer2, future_transfer
  input_importance_800/ leave-one-out input-feature ablation
  loss_ablation_800/    leave-one-out loss-component ablation
  transfer_ratio/       transfer-coverage experiments (1, 3, 8 and 15 tiles)

notebooks/      Jupyter notebooks for analysis                            [notebooks/README.md]
  results/              prediction, error maps and evaluation metrics
  experiments/          EDA (01), input importance (02), loss ablation (03),
                        transfer scaling (04); generated figures included

outputs/        configs, metrics.csv, test_results.json and checkpoints   [outputs/README.md]                                
```

## Key models (weights included)

| Model | Domain | Performance | Notes |
|---|---|---|---|
| **Temporal Generalisation** | full 800×800, 12 levels (15–195 m) | RMSE 0.755 / r 0.887 | Unseen time periods within the same domain |
| **Spatial Transfer 1** | held-out interior tile [200:400, 200:400] | RMSE 0.546 / r 0.511 | Transfer to an unseen neighbouring region |
| **Spatial Transfer 2** | held-out centre tile [0:200, 400:600] | RMSE 2.628 / r 0.713 | Transfer to an unseen neighbouring region |
| **Future Transfer** | full 800×800; train Jul+Aug, test held-out Sep | RMSE 1.068 / r 0.766 (Sep 4–6) | Extrapolation to a later, unseen month; whole-Sep RMSE 22.8 is driven by the anomalous 7 Sep |

Other runs' weights are not distributed (regenerate via their config);
`outputs/README.md` documents every experiment and its result.

## Model

- **Backbone**: 3D U-Net with anisotropic convolutions and horizontal-only pooling (vertical resolution preserved).
- **FiLM conditioning**: ERA5 winds / θ / qv / PM2.5 profiles + surface pressure + hour encoding modulate the feature maps after each convolution block through per-channel, per-level γ and β.
- **Bottleneck self-attention**: per-level spatial attention captures long-range spatial structure (the full-domain model uses an additional pooling stage to obtain a 50 × 50 bottleneck).
- **Input**: 18 channels = 9 physical variables × {t, t−1}.
- **Composite loss**: weighted MSE (peak up-weighting) + non-negativity + spatial anomaly + gradient + pattern-correlation + spectral consistency.

## Ablation studies

- **Input importance**: boundary PM2.5 and surface emissions are the dominant inputs overall. Horizontal winds mainly determine the spatial structure, particularly at higher levels, while the preceding-hour inputs become increasingly important aloft.
- **Loss components**: anomaly and gradient terms mainly improve prediction accuracy and local sharpness, whereas the pattern-correlation and spectral terms preserve spatial structure, particularly at higher levels. The non-negativity term provides a physical constraint with little effect on the reported metrics.

Both studies render directly from the JSON summaries in `outputs/ablation_{input,loss}/`, so the git-ignored ablation checkpoints are not required.

## Reproduce

```bash
pip install -r requirements.txt
# The original PALM simulation data are not included in this repository.
# point src/dataloader.py BASE_DIR at the PALM output tree, then:
python3 src/train.py --config configs/main/full_domain.yaml
```

## Data Availability

The PALM simulation data used in this project are too large to be uploaded to GitHub or publicly hosted online. If access to the original datasets is required, please contact the author.
