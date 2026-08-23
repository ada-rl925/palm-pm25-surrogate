"""
dataloader.py — PALMDatasetV2 with 3-D emission routing (9 input channels).

Surface emission is placed at the z level corresponding to the actual emission
surface (terrain + rooftop), rather than being broadcast uniformly across all
levels.  The three static channels (street_type, buildings_2d, zt) are removed.

Output shape of __getitem__
---------------------------
  x          (9, N_Z, H, W)  float32 — full 3-D input volume
  y          (N_Z,   H, W)   float32 — PM2.5 target (log1p → z-score)
  valid_mask (N_Z,   H, W)   bool    — True = outdoor cell

Channel layout of x  (N_CHANNELS = 9)
---------------------------------------
  ch 0   PM25 emission 3-D   log1p → z-score; emission value at emission surface,
                              background = 0 (legacy) or -mean/std (em_bg_norm=True)
                              surface height = zt + buildings_2d per pixel
                              emission_z = first Z level >= surface_height
                              (clamped to z=0 when surface_height < Z_HEIGHTS[0])
  ch 1   ls_forcing_left_u    z-score, per-level
  ch 2   ls_forcing_left_v    z-score, per-level
  ch 3   ls_forcing_left_pt   z-score, per-level
  ch 4   ls_forcing_left_qv   z-score, per-level
  ch 5   ls_forcing_left_PM25 log1p → z-score, per-level
  ch 6   surface_pressure     z-score, broadcast
  ch 7   hour_sin             no scaling, broadcast
  ch 8   hour_cos             no scaling, broadcast
"""


import os
import glob
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import netCDF4 as nc
import torch
from torch.utils.data import Dataset

# ── Simulation registry ───────────────────────────────────────────────────────
# Defaults to the compact dataset shipped in the repo (../data). Override with the
# PALM_DATA_DIR env var to point at a full raw PALM output tree instead.
BASE_DIR = os.environ.get(
    'PALM_DATA_DIR',
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'),
)

SIM_CONFIG = {
    'london_cm20190720-23': {
        'label':       'Jul',
        'chem_file':   'london_cm201907_chemistry',
        'static_file': 'london_cm201907_static',
        'dyn_file':    'london_cm201907_dynamic',
        'extra_out':   [],
        'sim_hours':   96,
    },
    'london_cm20190820-23': {
        'label':       'Aug',
        'chem_file':   'london_cm_d1_chemistry',
        'static_file': 'london_cm_d1_static',
        'dyn_file':    'london_cm_d1_dynamic',
        'extra_out':   [],
        'sim_hours':   96,
    },
    'london_cm20190904-07': {
        'label':       'Sep',
        'chem_file':   'london_cm201909_chemistry',
        'static_file': 'london_cm201909_static',
        'dyn_file':    'london_cm201909_dynamic',
        'extra_out':   ['london_cm20190908_OUTPUT', 'london_cm20190909_OUTPUT'],
        'sim_hours':   144,
    },
}

# ── Vertical level mapping ────────────────────────────────────────────────────
# ERA5 dynamic files have 80 levels (5 m – 1783 m).
# These 16 indices align exactly with PALM's output heights above ground level
# (the first PALM output level, ~5 m, is skipped via PALM_OUT_Z0).
PALM_Z_IDX   = [1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19, 29, 39, 59, 79]
N_Z          = len(PALM_Z_IDX)   # 16
PALM_OUT_Z0  = 1                 # skip first PALM output level (ground, ~5m)

# ── Feature definitions ───────────────────────────────────────────────────────
DYN_PROFILE_VARS = [
    'ls_forcing_left_u',
    'ls_forcing_left_v',
    'ls_forcing_left_pt',
    'ls_forcing_left_qv',
    'ls_forcing_left_PM25',
]
DYN_SCALAR_VARS = ['surface_forcing_surface_pressure']

DYN_PROFILE_DIM = len(DYN_PROFILE_VARS) * N_Z   # 85
DYN_SCALAR_DIM  = len(DYN_SCALAR_VARS)           # 1
DYN_TOTAL_DIM   = DYN_PROFILE_DIM + DYN_SCALAR_DIM  # 86

DYN_PM25_SLICE = slice(4 * N_Z, 5 * N_Z)  # ls_forcing_left_PM25 in dynamic vector

PM25_SCALE = 1e9  # kg m⁻³ → μg m⁻³


# ── Helper ────────────────────────────────────────────────────────────────────
def _parse_origin(ds) -> datetime:
    raw = getattr(ds, 'origin_time', '2019-01-01 00:00:00 +00')
    return datetime.strptime(raw.split(' +')[0].strip(), '%Y-%m-%d %H:%M:%S')


from model import Z_HEIGHTS   # (16,) float32 array of PALM output heights

N_CHANNELS = 9    # total input channels for this dataloader


# ── Dataset ───────────────────────────────────────────────────────────────────
class PALMDatasetV2(Dataset):
    """
    Parameters
    ----------
    sims           : list[str] | None           simulations to load; None = all three
    normalise      : bool                       z-score normalisation; call fit_normalisation() first
    roi            : (y0,y1,x0,x1) | None       spatial crop
    hour_filter    : dict[sim, set[int]] | None
    global_pm_norm : bool                       True = single mean/std; False = per-level
    z_range        : (z0, z1) | None            restrict output to levels [z0, z1)
    em_bg_norm     : bool                       True = z-score the full 3-D emission
                                                volume so background ("no emission
                                                here") and a zero-emission surface
                                                cell encode to the same value
                                                (-mean/std).  False = legacy
                                                behaviour: background stays 0,
                                                which collides with the encoding
                                                of an average-emission cell.
    t_hist         : int                        number of previous hours stacked
                                                into the input (temporal context).
                                                0 = current hour only (legacy,
                                                9 channels).  K > 0 gives
                                                9*(K+1) channels laid out as
                                                [em(t)..em(t-K)] followed by
                                                K+1 blocks of 8 conditioning
                                                channels (one block per hour).
                                                Hours before the simulation start
                                                are clamped to hour 1.
    prognostic     : bool                       Time-stepping mode: prepend the
                                                PM2.5 OUTPUT field at t-1 (the
                                                target variable's previous state)
                                                as an extra spatial channel, so
                                                the model predicts field(t) from
                                                field(t-1) + current forcing.
                                                At t=1 (no previous output) the
                                                boundary PM25 inflow profile is
                                                broadcast as a uniform initial
                                                field (no leak of interior state).
                                                Layout becomes
                                                [em(t)..em(t-K), pm25(t-1)] +
                                                K+1 conditioning blocks.
    """

    def __init__(
        self,
        sims:           Optional[List[str]]              = None,
        normalise:      bool                             = False,
        roi:            Optional[Tuple[int,int,int,int]] = None,
        hour_filter:    Optional[Dict[str, Set[int]]]    = None,
        global_pm_norm: bool                             = True,
        z_range:        Optional[Tuple[int, int]]        = None,
        em_bg_norm:     bool                             = False,
        t_hist:         int                              = 0,
        prognostic:     bool                             = False,
        zero_channels:  Optional[List[int]]              = None,
        patch_size:     Optional[int]                    = None,
        allowed_tiles:  Optional[List[Tuple[int,int]]]   = None,
        mask_region:    Optional[Tuple[int,int,int,int]]  = None,
        exclude_region: Optional[Tuple[int,int,int,int]]  = None,
        augment:        bool                             = False,
    ):
        self.sims           = sims or list(SIM_CONFIG.keys())
        self.normalise      = normalise
        self.global_pm_norm = global_pm_norm
        self.roi            = roi
        self.hour_filter    = hour_filter
        self.z_range        = z_range
        self.em_bg_norm     = em_bg_norm
        self.t_hist         = t_hist
        # input channels to constant-fill with 0 (= remove their information);
        # 0 is the mean of every z-scored channel, so this cleanly ablates them.
        self.zero_channels  = zero_channels or []
        # patch_size: if set, __getitem__ returns a random patch_size x patch_size
        # spatial crop (tile training on the full domain). None = no cropping.
        self.patch_size     = patch_size
        # allowed_tiles: if set (list of (y0,x0) top-left origins), the patch crop
        # is chosen uniformly from these fixed tiles instead of a free random crop
        # (used for spatial tile-holdout training). None = free random crop.
        self.allowed_tiles  = allowed_tiles
        # mask_region: (y0,y1,x0,x1) | None. Zero the emission (spatial) input
        # inside this region, so the model must predict that region's PM2.5 from
        # the surrounding tiles' context only (spatial "inpainting" study).
        self.mask_region    = mask_region
        # exclude_region: (y0,y1,x0,x1) | None. With patch_size and no
        # allowed_tiles, the free random crop rejects any window overlapping
        # this region (spatial holdout with arbitrary crop origins).
        self.exclude_region = exclude_region
        # augment: random horizontal mirror flips of the crop (train only).
        # Mirroring x negates physical u, mirroring y negates physical v; the
        # conditioning channels are z-scored, so the sign flip is applied in
        # physical space:  u'_norm = -u_norm - 2*mean/std.
        self.augment        = augment
        self.prognostic     = prognostic

        # Per-pixel z level at which the surface emission is injected. The compact
        # dataset ships this as a pre-baked field (emission_z.npy) so the OS-derived
        # static drivers need not be redistributed; if it is absent, fall back to
        # computing it from the raw static driver (buildings_2d + terrain).
        ez_path = os.path.join(BASE_DIR, 'emission_z.npy')
        if os.path.exists(ez_path):
            self._emission_z = np.load(ez_path).astype(np.int32)   # (800, 800)
        else:
            b2d_raw, zt_raw   = self._load_static_raw()   # (800, 800) each
            self._emission_z  = self._compute_emission_z(b2d_raw, zt_raw)  # (800, 800) int32

        self._emissions = {s: self._load_emissions(s) for s in self.sims}
        self._dynamics  = {s: self._load_dynamic(s)   for s in self.sims}
        self._out_lookup: Dict[Tuple[str, int], Tuple[str, int]] = {}
        self._index     = self._build_index()
        self._stats: Optional[dict] = None

    # ── Static ────────────────────────────────────────────────────────────────
    def _load_static_raw(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return raw (b2d, zt) arrays (800×800, fill→0)."""
        ref  = 'london_cm20190820-23'
        path = os.path.join(BASE_DIR, ref, 'INPUT', SIM_CONFIG[ref]['static_file'])
        with nc.Dataset(path) as ds:
            b2d = np.array(ds.variables['buildings_2d'][:], dtype=np.float32)
            fv  = float(ds.variables['buildings_2d']._FillValue)
            b2d[b2d == fv] = 0.0

            zt  = np.array(ds.variables['zt'][:], dtype=np.float32)
            fv  = getattr(ds.variables['zt'], '_FillValue', None)
            if fv is not None:
                zt[zt == float(fv)] = 0.0
        return b2d, zt

    def _compute_emission_z(self, b2d_raw: np.ndarray, zt_raw: np.ndarray) -> np.ndarray:
        """
        For each pixel (y,x), find the z level index where emission is placed.

        surface_height = zt_raw + b2d_raw
          (no building → pure terrain;  building → terrain + rooftop height)

        emission_z = first index i where Z_HEIGHTS[i] >= surface_height
        If surface_height < Z_HEIGHTS[0], clamp to z=0 (lowest available level).
        If surface_height > Z_HEIGHTS[-1], clamp to z=N_Z-1.
        """
        surface_h  = (zt_raw + b2d_raw).ravel()          # (800*800,)
        emission_z = np.searchsorted(Z_HEIGHTS, surface_h, side='left')  # first z >= surface_h
        emission_z = np.clip(emission_z, 0, N_Z - 1).reshape(b2d_raw.shape).astype(np.int32)
        return emission_z

    # ── Emissions ─────────────────────────────────────────────────────────────
    def _load_emissions(self, sim_name: str) -> np.ndarray:
        """Returns (T, 800, 800) raw PM25 values."""
        cfg  = SIM_CONFIG[sim_name]
        path = os.path.join(BASE_DIR, sim_name, 'INPUT', cfg['chem_file'])
        with nc.Dataset(path) as ds:
            t_hrs = np.array(ds.variables['time'][:])
            mask  = t_hrs <= cfg['sim_hours']
            names = [''.join(
                [c.decode() if isinstance(c, bytes) else ''
                 for c in row if not np.ma.is_masked(c)]
            ).strip() for row in ds.variables['emission_name'][:]]
            ev = np.ma.filled(ds.variables['emission_values'][mask],
                              fill_value=0.0).astype(np.float32)
        ev = ev[:, 0, :, :, :]
        return ev[:, :, :, names.index('PM25')]  # (T, Y, X)

    def _get_emission_at(self, sim_name: str, t_hour: int) -> np.ndarray:
        return self._emissions[sim_name][(t_hour - 1) % len(self._emissions[sim_name])].copy()

    # ── Dynamic ───────────────────────────────────────────────────────────────
    def _load_dynamic(self, sim_name: str) -> np.ndarray:
        """Returns (T, 86): 5 profiles × 16 PALM levels + surface_pressure."""
        cfg  = SIM_CONFIG[sim_name]
        path = os.path.join(BASE_DIR, sim_name, 'INPUT', cfg['dyn_file'])
        with nc.Dataset(path) as ds:
            profiles = [np.array(ds.variables[v][:, PALM_Z_IDX], dtype=np.float32)
                        for v in DYN_PROFILE_VARS]
            scalars  = [np.array(ds.variables[v][:], dtype=np.float32)
                        for v in DYN_SCALAR_VARS]
        return np.concatenate(
            profiles + [s[:, np.newaxis] for s in scalars], axis=1
        )  # (T, 86)

    def _get_dynamic_at(self, sim_name: str, t_hour: int) -> np.ndarray:
        dyn = self._dynamics[sim_name]
        return dyn[t_hour % len(dyn)].copy()

    # ── Time encoding ─────────────────────────────────────────────────────────
    @staticmethod
    def _time_enc(t: datetime) -> np.ndarray:
        h = t.hour + t.minute / 60.0
        return np.array([math.sin(2*math.pi*h/24),
                         math.cos(2*math.pi*h/24)], dtype=np.float32)

    # ── Index ─────────────────────────────────────────────────────────────────
    def _build_index(self) -> list:
        index = []
        for sim in self.sims:
            cfg   = SIM_CONFIG[sim]
            out   = os.path.join(BASE_DIR, sim, 'OUTPUT')
            files = sorted(glob.glob(os.path.join(out, '*_av_xy.*.nc')))
            for ext in cfg['extra_out']:
                files += sorted(glob.glob(
                    os.path.join(BASE_DIR, ext, '*_av_xy.*.nc')))
            if not files:
                raise FileNotFoundError(f'No _av_xy.nc for {sim}')
            with nc.Dataset(files[0]) as ds0:
                origin = _parse_origin(ds0)
            allowed = self.hour_filter.get(sim) if self.hour_filter else None
            for path in files:
                with nc.Dataset(path) as ds:
                    t_sec = np.array(ds.variables['time'][:])
                for t_idx, s in enumerate(t_sec):
                    t_h = round(float(s) / 3600)
                    # lookup over ALL hours (ignores split filter) so the
                    # prognostic mode can fetch t-1 even across splits
                    self._out_lookup[(sim, t_h)] = (path, t_idx)
                    if allowed is not None and t_h not in allowed:
                        continue
                    index.append((sim, path, t_idx, t_h, origin + timedelta(seconds=float(s))))
        return index

    # ── Output field loading (prognostic mode) ──────────────────────────────────
    def _load_output_field(self, sim: str, t_hour: int) -> Optional[np.ndarray]:
        """Normalised PM2.5 OUTPUT field (N_Z,H,W) at (sim, t_hour), or None if absent."""
        key = (sim, t_hour)
        if key not in self._out_lookup:
            return None
        path, t_idx = self._out_lookup[key]
        with nc.Dataset(path) as ds:
            pm_raw = np.array(ds.variables['kc_PM25_xy'][t_idx].filled(np.nan), dtype=np.float32)
        pm_raw = pm_raw[PALM_OUT_Z0:]
        m  = ~np.isnan(pm_raw)
        pm = np.log1p(np.where(m, pm_raw, 0.0) * PM25_SCALE)
        if self.roi is not None:
            y0, y1, x0, x1 = self.roi
            pm = pm[..., y0:y1, x0:x1]
        if self.normalise and self._stats is not None:
            s = self._stats
            pm = (pm - s['pm25']['mean'][:, np.newaxis, np.newaxis]) / s['pm25']['std'][:, np.newaxis, np.newaxis]
        return pm

    def _prev_field(self, sim: str, t_hour: int, H: int, W: int) -> np.ndarray:
        """Previous PM2.5 field for prognostic input; falls back to the boundary
        PM25 inflow profile (uniform, no interior leak) when t-1 is unavailable."""
        prev = self._load_output_field(sim, t_hour - 1)
        if prev is not None:
            return prev
        dyn  = self._get_dynamic_at(sim, t_hour)
        prof = np.log1p(dyn[DYN_PM25_SLICE] * PM25_SCALE)                 # (N_Z,)
        prev = np.broadcast_to(prof[:, np.newaxis, np.newaxis], (N_Z, H, W)).astype(np.float32)
        if self.normalise and self._stats is not None:
            s = self._stats
            prev = (prev - s['pm25']['mean'][:, np.newaxis, np.newaxis]) / s['pm25']['std'][:, np.newaxis, np.newaxis]
        return prev

    # ── Dataset interface ─────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        sim, av_path, t_idx, t_hour, t_abs = self._index[idx]

        # ── Target + mask ─────────────────────────────────────────────────────
        with nc.Dataset(av_path) as ds:
            pm_raw = np.array(
                ds.variables['kc_PM25_xy'][t_idx].filled(np.nan), dtype=np.float32)
        pm_raw     = pm_raw[PALM_OUT_Z0:]                   # (N_Z, 800, 800)
        valid_mask = ~np.isnan(pm_raw)
        pm25       = np.log1p(np.where(valid_mask, pm_raw, 0.0) * PM25_SCALE)

        # ── Spatial crop (target / mask / static emission routing) ───────────
        emission_z = self._emission_z   # (800, 800)
        if self.roi is not None:
            y0, y1, x0, x1 = self.roi
            pm25       = pm25      [..., y0:y1, x0:x1]
            valid_mask = valid_mask[..., y0:y1, x0:x1]
            emission_z = emission_z[     y0:y1, x0:x1]

        H, W = emission_z.shape
        s = self._stats if (self.normalise and self._stats is not None) else None

        if s is not None:
            nz_pm     = pm25.shape[0]   # target level count (may be < len(stats) when the
                                        # compact av_xy is pre-trimmed to the used levels)
            pm25_mean = s['pm25']['mean'][:nz_pm, np.newaxis, np.newaxis]  # (nz,1,1)
            pm25_std  = s['pm25']['std'][:nz_pm, np.newaxis, np.newaxis]
            pm25      = (pm25 - pm25_mean) / pm25_std

        # ── Per-hour feature blocks: t, t-1, ..., t-t_hist ───────────────────
        # x layout (9*(K+1) channels):
        #   [em(t) .. em(t-K)]  then  K+1 blocks of
        #   [u, v, pt, qv, PM25, surface_pressure, hour_sin, hour_cos]
        # For K=0 this reduces exactly to the legacy 9-channel layout.
        #
        # Emission 3-D channel: value placed at the emission surface z level.
        #   em_bg_norm=True : scatter raw log1p values, then z-score the whole
        #                     volume — background and zero-emission cells both
        #                     map to -mean/std (consistent encoding).
        #   em_bg_norm=False: legacy — em z-scored before scatter, background
        #                     stays 0, i.e. aliases an average-emission cell.
        rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

        def _p(a):  # (N_Z,) → (N_Z,H,W) same value at every (y,x) per level
            return np.broadcast_to(a[:, np.newaxis, np.newaxis], (N_Z, H, W))

        def _c(v):  # scalar → (N_Z,H,W)
            return np.full((N_Z, H, W), v, dtype=np.float32)

        em_blocks, cond_blocks = [], []
        for j in range(self.t_hist + 1):
            t_j = max(t_hour - j, 1)            # clamp before simulation start
            em  = np.log1p(self._get_emission_at(sim, t_j) * PM25_SCALE)
            dyn = self._get_dynamic_at(sim, t_j)
            dyn[DYN_PM25_SLICE] = np.log1p(dyn[DYN_PM25_SLICE] * PM25_SCALE)
            te  = self._time_enc(t_abs - timedelta(hours=t_hour - t_j))

            if self.roi is not None:
                y0, y1, x0, x1 = self.roi
                em = em[y0:y1, x0:x1]
            if s is not None:
                if not self.em_bg_norm:
                    em = (em - s['emission']['mean']) / s['emission']['std']
                dyn = (dyn - s['dynamic']['mean']) / s['dynamic']['std']

            em_3d = np.zeros((N_Z, H, W), dtype=np.float32)
            em_3d[emission_z, rows, cols] = em  # vectorised scatter
            if s is not None and self.em_bg_norm:
                em_3d = (em_3d - s['emission']['mean']) / s['emission']['std']

            em_blocks.append(em_3d)
            cond_blocks.append(np.stack([
                _p(dyn[0*N_Z : 1*N_Z]),                    # u
                _p(dyn[1*N_Z : 2*N_Z]),                    # v
                _p(dyn[2*N_Z : 3*N_Z]),                    # pt
                _p(dyn[3*N_Z : 4*N_Z]),                    # qv
                _p(dyn[4*N_Z : 5*N_Z]),                    # PM25
                _c(dyn[DYN_PROFILE_DIM + 0]),              # surface_pressure
                _c(te[0]),                                 # hour_sin
                _c(te[1]),                                 # hour_cos
            ], axis=0))

        # Prognostic mode: insert pm25(t-1) as a spatial channel right after the
        # emission block(s). Layout: [em(t)..em(t-K), pm25(t-1)] + cond blocks.
        spatial_parts = [np.stack(em_blocks, axis=0)]
        if self.prognostic:
            spatial_parts.append(self._prev_field(sim, t_hour, H, W)[np.newaxis])
        x = np.concatenate(
            spatial_parts + cond_blocks, axis=0
        ).astype(np.float32)                               # (C, N_Z, H, W)

        if self.zero_channels:
            x[self.zero_channels] = 0.0   # ablate listed input channels (slim-input study)

        if self.mask_region is not None:   # zero emission input in a region (spatial inpainting)
            y0, y1, x0, x1 = self.mask_region
            nsp = self.t_hist + 1          # emission spatial channels (em(t)..em(t-K))
            x[:nsp, :, y0:y1, x0:x1] = 0.0

        if self.patch_size is not None:   # spatial crop (tile training)
            ps = self.patch_size
            Hf, Wf = x.shape[-2], x.shape[-1]
            if self.allowed_tiles:                       # pick from fixed allowed tiles
                iy, ix = self.allowed_tiles[np.random.randint(len(self.allowed_tiles))]
                x          = x[...,          iy:iy+ps, ix:ix+ps]
                pm25       = pm25[...,       iy:iy+ps, ix:ix+ps]
                valid_mask = valid_mask[...,  iy:iy+ps, ix:ix+ps]
            elif Hf > ps or Wf > ps:                     # free random crop (legacy)
                for _ in range(200):
                    iy = np.random.randint(0, max(Hf - ps, 0) + 1)
                    ix = np.random.randint(0, max(Wf - ps, 0) + 1)
                    if self.exclude_region is None:
                        break
                    ey0, ey1, ex0, ex1 = self.exclude_region
                    if not (iy < ey1 and iy + ps > ey0 and ix < ex1 and ix + ps > ex0):
                        break                            # crop clear of excluded region
                x          = x[...,          iy:iy+ps, ix:ix+ps]
                pm25       = pm25[...,       iy:iy+ps, ix:ix+ps]
                valid_mask = valid_mask[...,  iy:iy+ps, ix:ix+ps]

        if self.augment:
            nsp = self.t_hist + 1 + (1 if self.prognostic else 0)  # spatial channels

            def _flip_wind(comp):   # comp 0=u, 1=v: physical sign flip in z-scored space
                for j in range(self.t_hist + 1):
                    c = nsp + 8 * j + comp
                    if s is not None:
                        dm = s['dynamic']['mean'][comp*N_Z:(comp+1)*N_Z]
                        dsd = s['dynamic']['std'][comp*N_Z:(comp+1)*N_Z]
                        x[c] = -x[c] - (2.0 * dm / dsd)[:, np.newaxis, np.newaxis]
                    else:
                        x[c] = -x[c]

            if np.random.rand() < 0.5:                   # mirror x (W axis): u -> -u
                x = x[..., ::-1].copy()
                pm25 = pm25[..., ::-1].copy()
                valid_mask = valid_mask[..., ::-1].copy()
                _flip_wind(0)
            if np.random.rand() < 0.5:                   # mirror y (H axis): v -> -v
                x = x[..., ::-1, :].copy()
                pm25 = pm25[..., ::-1, :].copy()
                valid_mask = valid_mask[..., ::-1, :].copy()
                _flip_wind(1)

        if self.z_range is not None:
            z0, z1 = self.z_range
            x          = x[:, z0:z1]
            pm25       = pm25[z0:z1]
            valid_mask = valid_mask[z0:z1]

        return (
            torch.from_numpy(x),                       # (9, N_Z, H, W)
            torch.from_numpy(pm25),                    # (N_Z, H, W)
            torch.from_numpy(valid_mask),              # (N_Z, H, W) bool
        )

    # ── Normalisation ─────────────────────────────────────────────────────────
    def fit_normalisation(self, indices: Optional[List[int]] = None):
        """Compute z-score stats from training samples (on log1p-transformed values)."""
        indices = indices or list(range(len(self)))
        print(f'Fitting normalisation on {len(indices)} samples...')

        em_sum, em_sq, em_n = 0.0, 0.0, 0
        dyn_sum = dyn_sq    = None
        pm_sum_g = 0.0;  pm_sq_g = 0.0;  pm_n_g = 0
        pm_sum_l = np.zeros(N_Z, np.float64)
        pm_sq_l  = np.zeros(N_Z, np.float64)
        pm_n_l   = np.zeros(N_Z, np.int64)
        nz_pm    = N_Z   # actual target level count (compact av_xy may hold fewer)

        for i, idx in enumerate(indices):
            sim, av_path, t_idx, t_hour, _ = self._index[idx]

            em  = np.log1p(self._get_emission_at(sim, t_hour) * PM25_SCALE)
            dyn = self._get_dynamic_at(sim, t_hour)
            dyn[DYN_PM25_SLICE] = np.log1p(dyn[DYN_PM25_SLICE] * PM25_SCALE)

            with nc.Dataset(av_path) as ds:
                pm_raw = np.array(
                    ds.variables['kc_PM25_xy'][t_idx].filled(np.nan), dtype=np.float32)
            pm_raw = pm_raw[PALM_OUT_Z0:]
            mask   = ~np.isnan(pm_raw)

            if self.roi is not None:
                y0, y1, x0, x1 = self.roi
                em     = em    [..., y0:y1, x0:x1]
                mask   = mask  [..., y0:y1, x0:x1]
                pm_raw = pm_raw[..., y0:y1, x0:x1]

            pm = np.log1p(np.where(mask, pm_raw, 0.0) * PM25_SCALE)
            nz_pm = pm.shape[0]

            flat = em.ravel().astype(np.float64)
            em_sum += flat.sum(); em_sq += (flat**2).sum(); em_n += flat.size

            if dyn_sum is None:
                dyn_sum = np.zeros(DYN_TOTAL_DIM, dtype=np.float64)
                dyn_sq  = np.zeros_like(dyn_sum)
            dyn_sum += dyn.astype(np.float64)
            dyn_sq  += dyn.astype(np.float64) ** 2

            if self.global_pm_norm:
                valid_pm = pm[mask].astype(np.float64)
                pm_sum_g += valid_pm.sum()
                pm_sq_g  += (valid_pm ** 2).sum()
                pm_n_g   += valid_pm.size
            else:
                for z in range(nz_pm):
                    v = pm[z][mask[z]].astype(np.float64)
                    pm_sum_l[z] += v.sum()
                    pm_sq_l[z]  += (v ** 2).sum()
                    pm_n_l[z]   += v.size

            if (i + 1) % 50 == 0:
                print(f'  {i+1}/{len(indices)}')

        def _std(sq, s, n):
            return float(np.sqrt(max(sq/n - (s/n)**2, 1e-8)))

        N = len(indices)
        if self.global_pm_norm:
            pm25_mean = np.full(nz_pm, pm_sum_g / max(pm_n_g, 1), dtype=np.float32)
            pm25_std  = np.full(nz_pm, _std(pm_sq_g, pm_sum_g, pm_n_g), dtype=np.float32)
        else:
            pm25_mean = np.array([pm_sum_l[z] / max(pm_n_l[z], 1) for z in range(nz_pm)], dtype=np.float32)
            pm25_std  = np.array([_std(pm_sq_l[z], pm_sum_l[z], pm_n_l[z]) for z in range(nz_pm)], dtype=np.float32)

        self._stats = {
            'emission': {
                'mean': np.float32(em_sum / em_n),
                'std':  np.float32(_std(em_sq, em_sum, em_n)),
            },
            'dynamic': {
                'mean': (dyn_sum / N).astype(np.float32),
                'std':  np.array([_std(dyn_sq[j], dyn_sum[j], N)
                                  for j in range(DYN_TOTAL_DIM)], dtype=np.float32),
            },
            'pm25': {'mean': pm25_mean, 'std': pm25_std},
        }
        print('Done.')
        print(f'  emission  mean={self._stats["emission"]["mean"]:.4f}  '
              f'std={self._stats["emission"]["std"]:.4f}')
        if self.global_pm_norm:
            print(f'  pm25      global mean={pm25_mean[0]:.4e}  std={pm25_std[0]:.4e}')
        else:
            print(f'  pm25      per-level  mean[0]={pm25_mean[0]:.4e}  mean[-1]={pm25_mean[-1]:.4e}')

    def get_stats(self) -> Optional[dict]:
        return self._stats

    # ── Utilities ─────────────────────────────────────────────────────────────
    @property
    def z_levels(self) -> np.ndarray:
        out = os.path.join(BASE_DIR, self.sims[0], 'OUTPUT')
        f   = sorted(glob.glob(os.path.join(out, '*_av_xy.*.nc')))[0]
        with nc.Dataset(f) as ds:
            return np.array(ds.variables['zu_xy'][PALM_OUT_Z0:])

    def __repr__(self) -> str:
        roi = self.roi or (0, 800, 0, 800)
        H, W = roi[1]-roi[0], roi[3]-roi[2]
        n_ch = N_CHANNELS * (self.t_hist + 1)
        return (f'PALMDatasetV2(sims={[SIM_CONFIG[s]["label"] for s in self.sims]}, '
                f'n={len(self)}, x=({n_ch},{N_Z},{H},{W}), y=({N_Z},{H},{W}))')


# ── Temporal split ────────────────────────────────────────────────────────────
def temporal_split(
    sims:           Optional[List[str]]                  = None,
    val_hours:      Optional[Dict[str, Tuple[int,int]]]  = None,
    test_hours:     Optional[Dict[str, Tuple[int,int]]]  = None,
    global_pm_norm: bool                                 = True,
    **dataset_kwargs,
) -> Tuple[PALMDatasetV2, PALMDatasetV2, PALMDatasetV2]:
    """Time-based train/val/test split using PALMDatasetV2 (9-channel emission routing)."""
    sims       = sims       or list(SIM_CONFIG.keys())
    val_hours  = val_hours  or {}
    test_hours = test_hours or {}

    def _filter(split):
        hf = {}
        for sim in sims:
            all_h  = set(range(1, SIM_CONFIG[sim]['sim_hours'] + 1))
            val_h  = set(range(*val_hours[sim]))  if sim in val_hours  else set()
            test_h = set(range(*test_hours[sim])) if sim in test_hours else set()
            hf[sim] = (all_h - val_h - test_h if split == 'train'
                       else val_h if split == 'val' else test_h)
        return hf

    tr = PALMDatasetV2(sims=sims, hour_filter=_filter('train'), global_pm_norm=global_pm_norm, **dataset_kwargs)
    va = PALMDatasetV2(sims=sims, hour_filter=_filter('val'),   global_pm_norm=global_pm_norm, **dataset_kwargs)
    te = PALMDatasetV2(sims=sims, hour_filter=_filter('test'),  global_pm_norm=global_pm_norm, **dataset_kwargs)
    print(f'Temporal split:  train={len(tr)}  val={len(va)}  test={len(te)}')
    return tr, va, te
