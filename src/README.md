# `src/` — source code

Core library for the PALM PM2.5 surrogate. All notebooks and configs import from here.

| File | Contents |
|------|----------|
| `model.py` | `UNet3DFiLM` — 3-D U-Net with FiLM conditioning, bottleneck self-attention, and an internal per-level height encoding; `PhysicsLoss` — weighted MSE + non-negativity + anomaly + gradient + pattern-correlation + spectral terms. |
| `dataloader.py` | `PALMDatasetV2` — builds the 18-channel input (9 drivers × {t, t−1}) and the 12-level hourly-mean PM2.5 target from the PALM NetCDF tree; `temporal_split` — train/test split by episode day. `BASE_DIR` points at the raw PALM output tree (edit for your machine). |
| `train.py` | Training entry point: `python3 src/train.py --config <yaml> [--resume <ckpt>]`. Reads a config, trains, writes `checkpoints/`, `metrics.csv`, `test_results.json` under the config's `output_dir`. Supports `zero_channels` (input ablation) and `zero_height` (height ablation). |
| `utils.py` | Small helpers (logging, seeding, metrics). |
| `draw_study_area_map.py` | **Standalone** figure helper (not part of training). Needs internet + `osmnx contextily geopandas`; regenerates the study-area maps **and the transfer-experiment layout figure** (`transfer_layouts.png`, train vs test tiles on the OSM basemap) into `notebooks/experiments/figs/`. Figures are already committed, so this is not needed to reproduce the analysis. |

## Model at a glance
- Input `X`: `(18, 12, H, W)` — 9 physical variables (emission, u, v, pt, qv, boundary PM2.5, surface pressure, hour sin/cos) each at hour *t* and *t−1*.
- Target `y`: `(12, H, W)` — PALM hourly-mean PM2.5, levels 15–195 m (z0–z11).
- ~1.6 M parameters; `n_down=4` (bottleneck 50×50) for the full 800×800 domain, `n_down=3` for 200×200 transfer tiles.
