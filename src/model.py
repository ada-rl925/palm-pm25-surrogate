"""
UNet3D-FiLM surrogate model + loss functions for PALM PM2.5 prediction.

Input  (B, 12, N_Z, H, W)
Output (B, N_Z, H, W)
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# PALM output heights (m) after skipping ground level (PALM_OUT_Z0 = 1)
Z_HEIGHTS = np.array([
    15., 25., 35., 45., 55., 75., 95., 115.,
    135., 155., 175., 195., 295., 419.86, 878.28, 1783.20,
], dtype=np.float32)


# ── Building blocks ───────────────────────────────────────────────────────────

class AsymConv3D(nn.Module):
    """(1,3,3) spatial conv → (3,1,1) vertical conv, each with GroupNorm + ReLU."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        g = min(8, out_ch)
        self.spatial = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, (1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.GroupNorm(g, out_ch),
            nn.ReLU(inplace=True),
        )
        layers = [
            nn.Conv3d(out_ch, out_ch, (3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.GroupNorm(g, out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout3d(dropout))
        self.vertical = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vertical(self.spatial(x))


class Down3D(nn.Module):
    """MaxPool xy by 2 (z unchanged) → AsymConv3D."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool3d((1, 2, 2)),
            AsymConv3D(in_ch, out_ch, dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up3D(nn.Module):
    """Upsample xy by 2 (z unchanged), concat skip, AsymConv3D."""

    def __init__(self, up_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv = AsymConv3D(up_ch + skip_ch, out_ch, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=(1., 2., 2.),
                          mode='trilinear', align_corners=True)
        return self.conv(torch.cat([skip, x], dim=1))


COND_DIM = 8   # 5 ERA5 profiles + 1 pressure + 2 time per z level


def _sinusoidal_2d(h: int, w: int, dim: int, device, dtype) -> torch.Tensor:
    """Standard 2-D sinusoidal positional encoding, (h*w, dim).

    Half the channels encode the row coordinate, half the column. Gives the
    bottleneck attention an absolute sense of geometry so it can synthesise
    oriented, domain-spanning structures (the upper-level streaks) rather than
    translation-invariant blobs.
    """
    d = dim // 4
    yy, xx = torch.meshgrid(torch.arange(h, device=device, dtype=dtype),
                            torch.arange(w, device=device, dtype=dtype),
                            indexing='ij')
    freq = torch.exp(torch.arange(d, device=device, dtype=dtype) * (-math.log(10000.0) / max(d, 1)))
    out = []
    for coord in (yy.reshape(-1, 1), xx.reshape(-1, 1)):
        ang = coord * freq[None, :]
        out += [ang.sin(), ang.cos()]
    pe = torch.cat(out, dim=1)                                  # (h*w, 4*d)
    if pe.shape[1] < dim:                                       # pad if dim not divisible by 4
        pe = torch.cat([pe, torch.zeros(h * w, dim - pe.shape[1], device=device, dtype=dtype)], dim=1)
    return pe


class BottleneckAttention(nn.Module):
    """Per-z-level spatial multi-head self-attention over the bottleneck grid.

    Treats each z level's (H*W) cells as a token sequence (channels = feature
    dim) so every location can attend to every other — letting the network
    build long-range coherent structure that local convolutions cannot. The
    residual is gated by a zero-initialised scalar, so the layer starts as an
    identity and learns its contribution gradually (stable with few samples).
    """

    def __init__(self, ch: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm  = nn.LayerNorm(ch)
        self.attn  = nn.MultiheadAttention(ch, n_heads, dropout=dropout, batch_first=True)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, Z, H, W = x.shape
        t  = x.permute(0, 2, 3, 4, 1).reshape(B * Z, H * W, C)  # (B*Z, H*W, C)
        pe = _sinusoidal_2d(H, W, C, x.device, x.dtype)[None]    # (1, H*W, C)
        tn = self.norm(t + pe)
        a, _ = self.attn(tn, tn, tn, need_weights=False)
        t = t + self.gamma * a
        return t.reshape(B, Z, H, W, C).permute(0, 4, 1, 2, 3)


# ── FiLM layer ────────────────────────────────────────────────────────────────

class FiLMLayer(nn.Module):
    """
    Generates per-channel, per-z-level scale γ and shift β from a conditioning
    vector and applies them to a 3D feature map.

    Initialised to identity (zero weights/bias → γ=0, β=0 → output = input).
    """

    def __init__(self, cond_dim: int, feat_ch: int):
        super().__init__()
        self.mlp = nn.Linear(cond_dim, 2 * feat_ch)
        nn.init.zeros_(self.mlp.weight)
        nn.init.zeros_(self.mlp.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x    : (B, C, N_Z, H, W)
        # cond : (B, N_Z, COND_DIM)
        C      = x.shape[1]
        params = self.mlp(cond)                                     # (B, N_Z, 2C)
        gamma  = params[..., :C].permute(0, 2, 1)[..., None, None]  # (B, C, N_Z, 1, 1)
        beta   = params[..., C:].permute(0, 2, 1)[..., None, None]
        return x * (1 + gamma) + beta                               # residual FiLM


# ── Model ─────────────────────────────────────────────────────────────────────

class UNet3DFiLM(nn.Module):
    """
    3-D U-Net with FiLM conditioning.

    Parameters
    ----------
    in_ch      : total input channels (18 = 9 drivers × {t, t-1})
    spatial_ch : channels fed to the encoder (2 = emission at t, t-1);
                 the remaining channels only condition the FiLM layers
    base_ch    : UNet base width
    dropout    : dropout in conv blocks
    n_z        : number of active z levels (12 for the final z0–11 config)
    z_start    : first z level index (for height encoding offset)
    """

    def __init__(
        self,
        in_ch:      int   = 12,
        spatial_ch: int   = 4,
        base_ch:    int   = 32,
        dropout:    float = 0.0,
        n_z:        int   = 16,
        z_start:    int   = 0,
        use_attention: bool = False,
        attn_heads:    int  = 4,
        n_down:        int  = 3,
    ):
        super().__init__()
        self.n_z        = n_z
        self.spatial_ch = spatial_ch
        self.n_down     = n_down   # 3 = legacy; 4 adds one pooling level (smaller bottleneck)
        # one 8-ch conditioning block per time step (1 for legacy layouts)
        self.n_blocks   = max((in_ch - spatial_ch) // COND_DIM, 1)
        cond_dim        = COND_DIM * self.n_blocks
        b = base_ch

        # height encoding (same log1p z-score as UNet3D)
        z = np.log1p(Z_HEIGHTS[z_start:z_start + n_z])
        z = (z - z.mean()) / (z.std() + 1e-8)
        self.register_buffer(
            'height_enc',
            torch.tensor(z, dtype=torch.float32).view(1, 1, n_z, 1, 1),
        )

        spatial_in = spatial_ch + 1   # +1 height channel

        # Encoder
        self.inc   = AsymConv3D(spatial_in, b,     dropout)
        self.film0 = FiLMLayer(cond_dim, b)
        self.down1 = Down3D(b,     b * 2, dropout)
        self.film1 = FiLMLayer(cond_dim, b * 2)
        self.down2 = Down3D(b * 2, b * 4, dropout)
        self.film2 = FiLMLayer(cond_dim, b * 4)
        self.down3 = Down3D(b * 4, b * 8, dropout)
        self.film3 = FiLMLayer(cond_dim, b * 8)

        # Optional extra pooling level (n_down=4): bottleneck becomes b*16 at H/16.
        # Used at full 800x800 so the bottleneck grid (e.g. 50x50) is small enough
        # for global attention. Only created for n_down>=4 -> legacy checkpoints
        # (n_down=3) keep identical state-dict keys and still load.
        if n_down >= 4:
            self.down4    = Down3D(b * 8, b * 16, dropout)
            self.film_d4  = FiLMLayer(cond_dim, b * 16)
            bott_ch = b * 16
        else:
            self.down4 = self.film_d4 = None
            bott_ch = b * 8
        self.attn  = BottleneckAttention(bott_ch, attn_heads, dropout) if use_attention else None
        if n_down >= 4:
            self.up0     = Up3D(b * 16, b * 8, b * 8, dropout)
            self.film_u0 = FiLMLayer(cond_dim, b * 8)
        else:
            self.up0 = self.film_u0 = None

        # Decoder
        self.up1   = Up3D(b * 8, b * 4, b * 4, dropout)
        self.film4 = FiLMLayer(cond_dim, b * 4)
        self.up2   = Up3D(b * 4, b * 2, b * 2, dropout)
        self.film5 = FiLMLayer(cond_dim, b * 2)
        self.up3   = Up3D(b * 2, b,     b,     dropout)
        self.film6 = FiLMLayer(cond_dim, b)

        self.head  = nn.Conv3d(b, 1, kernel_size=1)

    def _extract_cond(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract (B, N_Z, COND_DIM * n_blocks) conditioning tensor.

        Each 8-channel block after the spatial channels holds, for one hour:
          +0 .. +4  : 5 ERA5 profiles, per-level (same value at all H,W)
          +5        : surface_pressure (same at all z,H,W)
          +6, +7    : hour_sin, hour_cos (same at all z,H,W)
        Values are read at spatial position (0,0) — identical everywhere.
        """
        n_z   = self.n_z
        feats = []
        for j in range(self.n_blocks):
            base = self.spatial_ch + j * COND_DIM
            era5 = x[:, base:base+5, :, 0, 0].permute(0, 2, 1)            # (B, N_Z, 5)
            pres = x[:, base+5, 0, 0, 0].unsqueeze(1).unsqueeze(2)         # (B, 1, 1)
            pres = pres.expand(-1, n_z, 1)                                  # (B, N_Z, 1)
            time = x[:, base+6:base+8, 0, 0, 0].unsqueeze(1).expand(-1, n_z, -1)
            feats += [era5, pres, time]
        return torch.cat(feats, dim=-1)              # (B, N_Z, 8 * n_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, Z, H, W = x.shape

        cond   = self._extract_cond(x)                        # (B, N_Z, 8)
        x_spat = x[:, :self.spatial_ch]                       # (B, spatial_ch, N_Z, H, W)
        h_enc  = self.height_enc.expand(B, -1, Z, H, W)
        x_in   = torch.cat([x_spat, h_enc], dim=1)            # (B, spatial_ch+1, N_Z, H, W)

        x1 = self.film0(self.inc(x_in),    cond)
        x2 = self.film1(self.down1(x1),    cond)
        x3 = self.film2(self.down2(x2),    cond)
        x4 = self.film3(self.down3(x3),    cond)

        if self.down4 is not None:                 # extra pooling level (n_down=4)
            x5 = self.film_d4(self.down4(x4), cond)
            if self.attn is not None:
                x5 = self.attn(x5)
            y  = self.film_u0(self.up0(x5, x4), cond)
        else:
            if self.attn is not None:
                x4 = self.attn(x4)
            y  = x4

        y  = self.film4(self.up1(y,  x3),  cond)
        y  = self.film5(self.up2(y,  x2),  cond)
        y  = self.film6(self.up3(y,  x1),  cond)

        return self.head(y).squeeze(1)


# ── Loss ──────────────────────────────────────────────────────────────────────

def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not mask.any():
        return pred.new_tensor(0.0)
    return F.mse_loss(pred[mask], target[mask])


class PhysicsLoss(nn.Module):
    """Masked MSE + non-negativity + spatial anomaly + gradient + pattern-correlation + spectral.

    Non-negativity: if per-level norm stats (pm_mean, pm_std) are given, the
    penalty is applied to the de-normalised log1p concentration (which is the
    actual physical constraint: log1p(c) >= 0).  Without stats it falls back to
    the legacy normalised-space penalty, which also punishes valid below-mean
    predictions and biases the model upward — pass stats whenever possible.
    """

    def __init__(self, lambda_nonneg: float = 0.1,
                 lambda_anom: float = 0.0, lambda_grad: float = 0.0,
                 lambda_pattern: float = 0.0, pattern_min_std: float = 0.01,
                 lambda_spectral: float = 0.0, high_weight: float = 0.0,
                 pm_mean=None, pm_std=None):
        super().__init__()
        self.lw_nn   = lambda_nonneg
        self.lw_anom = lambda_anom
        self.lw_grad = lambda_grad
        self.lw_pat  = lambda_pattern
        self.pat_min_std = pattern_min_std
        self.lw_spec = lambda_spectral
        # high_weight>0: weight the main MSE by (1 + high_weight*relu(target)),
        # i.e. up-weight above-level-mean (high-concentration) pixels so the
        # model stops smoothing the peaks (counters regression-to-mean).
        self.high_weight = high_weight
        if pm_mean is not None:
            self.register_buffer(
                'pm_mean', torch.as_tensor(np.asarray(pm_mean), dtype=torch.float32).view(1, -1, 1, 1))
            self.register_buffer(
                'pm_std', torch.as_tensor(np.asarray(pm_std), dtype=torch.float32).view(1, -1, 1, 1))
        else:
            self.pm_mean = self.pm_std = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        if self.high_weight > 0 and mask.any():
            w  = 1.0 + self.high_weight * F.relu(target)
            se = (pred - target) ** 2
            l_main = (w * se)[mask].sum() / w[mask].sum()
        else:
            l_main = masked_mse(pred, target, mask)
        if self.pm_mean is not None and pred.ndim == 4:
            log_pred = pred * self.pm_std + self.pm_mean   # de-normalised log1p space
            l_nonneg = (F.relu(-log_pred)[mask].mean() if mask.any()
                        else pred.new_tensor(0.0))
        else:
            l_nonneg = F.relu(-pred - 0.1).mean()

        if pred.ndim == 4 and self.lw_anom > 0:
            mask_f     = mask.float()
            mask_sum   = mask_f.sum((-2, -1), keepdim=True).clamp(min=1)
            truth_mean = (target * mask_f).sum((-2, -1), keepdim=True) / mask_sum
            pred_mean  = (pred   * mask_f).sum((-2, -1), keepdim=True) / mask_sum
            l_anom     = masked_mse(pred - pred_mean, target - truth_mean, mask)
        else:
            l_anom = pred.new_tensor(0.0)

        if pred.ndim == 4 and self.lw_grad > 0:
            # x-direction gradient (W-1 columns)
            pred_dx  = pred[:, :, :, 1:] - pred[:, :, :, :-1]
            truth_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
            mask_dx  = mask[:, :, :, 1:] & mask[:, :, :, :-1]
            # y-direction gradient (H-1 rows)
            pred_dy  = pred[:, :, 1:, :] - pred[:, :, :-1, :]
            truth_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
            mask_dy  = mask[:, :, 1:, :] & mask[:, :, :-1, :]
            l_grad = (masked_mse(pred_dx, truth_dx, mask_dx) +
                      masked_mse(pred_dy, truth_dy, mask_dy))
        else:
            l_grad = pred.new_tensor(0.0)

        if pred.ndim == 4 and self.lw_pat > 0:
            # Scale-invariant spatial-structure term: per (sample, level),
            # 1 - Pearson correlation between the within-hour spatial anomalies
            # of pred and target. Amplitude-blind, so faint upper-level texture
            # counts as much as strong surface structure. Levels whose truth
            # anomaly std is below pat_min_std (physically near-uniform, e.g.
            # z14-15) carry no real structure and are excluded.
            mf  = mask.float()
            n   = mf.sum((-2, -1)).clamp(min=1)                       # (B, Z)
            pm  = (pred   * mf).sum((-2, -1)) / n
            tm  = (target * mf).sum((-2, -1)) / n
            pa  = (pred   - pm[..., None, None]) * mf
            ta  = (target - tm[..., None, None]) * mf
            cov   = (pa * ta).sum((-2, -1)) / n
            var_p = (pa ** 2).sum((-2, -1)) / n
            var_t = (ta ** 2).sum((-2, -1)) / n
            # eps INSIDE the sqrt: sqrt has an infinite gradient at 0, so a
            # near-uniform patch (var ~ 0) would back-propagate NaN. Adding eps
            # before the sqrt keeps the gradient bounded. (Clamping after sqrt
            # does NOT work: 0*inf from the clamp boundary is still NaN.)
            corr  = cov / ((var_p + 1e-6).sqrt() * (var_t + 1e-6).sqrt())
            w     = (var_t.sqrt() > self.pat_min_std).float()
            l_pat = ((1.0 - corr) * w).sum() / w.sum().clamp(min=1)
        else:
            l_pat = pred.new_tensor(0.0)

        if pred.ndim == 4 and self.lw_spec > 0:
            # Spectral term: match the per-level 2-D power spectrum of the
            # spatial anomaly. Amplitude/scale-statistics oriented (texture
            # realism: streak width, orientation, energy across scales) but
            # phase-blind, so it complements the pattern term. Invalid cells
            # are filled with the level mean before the FFT; levels below
            # pat_min_std (near-uniform) are excluded.
            mf  = mask.float()
            n   = mf.sum((-2, -1)).clamp(min=1)
            pm  = (pred * mf).sum((-2, -1)) / n
            tm  = (target * mf).sum((-2, -1)) / n
            pa  = (pred   - pm[..., None, None]) * mf
            ta  = (target - tm[..., None, None]) * mf
            Pp  = torch.fft.rfft2(pa).abs()
            Pt  = torch.fft.rfft2(ta).abs()
            l_map = (torch.log1p(Pp) - torch.log1p(Pt)).abs().mean((-2, -1))   # (B, Z)
            ts  = ((ta ** 2).sum((-2, -1)) / n).sqrt()
            w   = (ts > self.pat_min_std).float()
            l_spec = (l_map * w).sum() / w.sum().clamp(min=1)
        else:
            l_spec = pred.new_tensor(0.0)

        total = (l_main + self.lw_nn * l_nonneg
                 + self.lw_anom * l_anom + self.lw_grad * l_grad
                 + self.lw_pat * l_pat + self.lw_spec * l_spec)
        return total, {
            'total':  total.item(),
            'main':   l_main.item(),
            'nonneg': l_nonneg.item(),
            'anom':   l_anom.item(),
            'grad':   l_grad.item(),
            'pattern': l_pat.item(),
            'spectral': l_spec.item(),
        }
