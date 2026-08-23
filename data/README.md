# `data/` — compact PALM dataset for the surrogate

A **compact copy of the PALM LES data the surrogate reads** (~4.2 GB, vs ~79 GB of raw
output).

> **Zenodo:** DOI [10.5281/zenodo.21969523](https://doi.org/10.5281/zenodo.21969523)

## What's inside

Three PALM episodes over Camden, London (800×800 @ 10 m):

| Directory | Episode | Sim hours |
|-----------|---------|-----------|
| `london_cm20190720-23/` | 20–23 Jul 2019 | 96 |
| `london_cm20190820-23/` | 20–23 Aug 2019 | 96 |
| `london_cm20190904-07/` (+ `london_cm20190908_OUTPUT/`, `london_cm20190909_OUTPUT/`) | 04–09 Sep 2019 | 144 |

Each episode has:
- `INPUT/…_dynamic`   — ERA5 meteorology + CAMS chemical boundary profiles
- `INPUT/…_chemistry` — surface emissions (PM2.5 only)
- `OUTPUT/…_av_xy.*.nc` — hourly-mean PM2.5 target field (`kc_PM25_xy`)

plus one domain-wide file:
- `emission_z.npy` — per-cell z level at which the surface emission is injected (derived;
  see **Data sources**). Replaces the static driver the model would otherwise read.

## How it was slimmed (losslessly, for this model)

Derived from the raw PALM output tree — only data the model never reads is dropped, then
everything is zlib-compressed:

- **av_xy targets** — keep only `kc_PM25_xy`, `zu_xy`, `time`; keep only the 13 z-levels the
  model uses (ground + 15–195 m); drop all other output. (4.7 GB → ~0.3 GB each)
- **chemistry** — keep only the PM2.5 species and `t ≤ sim_hours`; emissions are ~98 % zero
  so they compress hard. (3.7 GB → ~0.03 GB each)
- **dynamic** — copied verbatim.
- **static drivers** — not redistributed; the only quantity the model uses from them (the
  per-cell surface-emission z level) is pre-baked into `emission_z.npy`. The building mask is
  taken directly from the PALM output (its NaN pattern), not from the static driver.

Numerically identical to reading the raw output: a per-element check of `X`, `y` and the mask
gives `max|Δ| = 0` on every episode.

## Data sources

Each PALM run is driven by **three input drivers**, following the driver-construction workflow
of the PALM CO₂ framework of [Li et al. (2026)](https://doi.org/10.5194/gmd-19-6417-2026):

- **Static** (time-invariant urban topography) — terrain height from
  [OS Terrain 50](https://www.ordnancesurvey.co.uk/products/os-terrain-50) and
  [1 m National LiDAR](https://environment.data.gov.uk/dataset/13787b9a-26a4-4775-8523-806d13af58fc);
  building shape & height from [OS MasterMap Topography](https://www.ordnancesurvey.co.uk/products/os-mastermap-topography-layer)
  and [OS Building Height Attribute](https://www.ordnancesurvey.co.uk/products/os-mastermap-building-height-attribute);
  land-use / surface-cover from [ESA WorldCover 10 m](https://doi.org/10.5281/zenodo.7254221).
- **Dynamic** (time-varying forcing, 80 vertical levels) — initial and boundary conditions from
  [ERA5 meteorology](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels)
  and [CAMS chemical background/boundary profiles](https://ads.atmosphere.copernicus.eu/datasets/cams-europe-air-quality-reanalyses).
- **Chemistry** (time-varying surface emissions) — an hourly, near-surface area source of five
  species that reduce to two independent inputs (PM2.5 and NOₓ), from
  [NAEI](https://naei.energysecurity.gov.uk/) and [CAMS-REG-TEMPO](https://doi.org/10.5194/essd-13-367-2021).

**This dataset contains only the subset needed to train and reproduce the surrogate — compressed.**
Of the full drivers above, what is redistributed here is:

| In this dataset | From driver | Source |
|---|---|---|
| `OUTPUT/…_av_xy.nc` — PM2.5 target | PALM output | PALM-4U LES (this work) |
| `INPUT/…_dynamic` — winds, θ, qv, pressure + PM2.5 boundary | dynamic | ERA5 + CAMS |
| `INPUT/…_chemistry` — PM2.5 emission | chemistry | NAEI + CAMS-REG-TEMPO |
| `emission_z.npy` — surface-emission z level (derived) | static | derived from OS + National LiDAR |

The static land-use/vegetation (ESA WorldCover) and the OS building/terrain products drive PALM
but are **not** surrogate inputs and are **not redistributed**; the only static quantity the
model uses — the per-cell surface-emission z level — is shipped as the derived `emission_z.npy`.
See **Licences**.

## Licences

Released under CC-BY 4.0. Sources: ERA5, CAMS European air-quality reanalysis, and
CAMS-REG-TEMPO ([10.5281/zenodo.15011342](https://doi.org/10.5281/zenodo.15011342)) under
CC-BY 4.0; NAEI and the Environment Agency National LiDAR Programme under the Open Government
Licence v3.0 — *Contains public sector information licensed under the Open Government Licence
v3.0.*

Ordnance Survey MasterMap Topography and Building Height Attribute are licensed via EDINA
Digimap and are not redistributed; `emission_z.npy` is a derived 10 m integer level index
containing no OS identifiers or vector geometry.

## Pointing the loader elsewhere

`BASE_DIR` defaults to this folder. To read a full raw PALM tree instead (with the original
static drivers, from which `emission_z` is recomputed automatically):

```bash
export PALM_DATA_DIR=/path/to/raw/palm/output
```
