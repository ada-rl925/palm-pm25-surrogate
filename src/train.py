"""
Training script for UNet3DFiLM — 3D volumetric PM2.5 prediction.

The lowest 12 z-levels (15–195 m) are predicted simultaneously.

Usage:
    python3 src/train.py --config configs/main/full_domain.yaml
    python3 src/train.py --config configs/main/full_domain.yaml --resume outputs/full_domain/checkpoints/last.pt
"""

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from dataloader import temporal_split, N_Z
from model import PhysicsLoss, masked_mse, UNet3DFiLM
from utils import CSVLogger


# ── Config / seed ─────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Data ──────────────────────────────────────────────────────────────────────

def build_datasets(cfg: dict):
    dcfg       = cfg['data']
    roi        = tuple(dcfg['roi']) if dcfg.get('roi') else None
    val_hours  = {k: tuple(v) for k, v in dcfg.get('val_hours', {}).items()}
    test_hours = {k: tuple(v) for k, v in dcfg['test_hours'].items()}
    z_range    = tuple(dcfg['z_range']) if dcfg.get('z_range') else None

    split_fn = temporal_split
    split_kw = dict(val_hours=val_hours, test_hours=test_hours,
                    roi=roi, normalise=True,
                    global_pm_norm=dcfg.get('global_pm_norm', True),
                    z_range=z_range,
                    em_bg_norm=dcfg.get('em_bg_norm', False),
                    t_hist=dcfg.get('t_hist', 0),
                    prognostic=dcfg.get('prognostic', False),
                    zero_channels=dcfg.get('zero_channels', None),
                    mask_region=tuple(dcfg['mask_region']) if dcfg.get('mask_region') else None)

    # ── spatial tile-holdout (additive; only when data.tile_size is set) ──
    # Train on the full domain but restrict the patch crop to a fixed 4x4 grid of
    # `tile_size` tiles, optionally excluding one (train_exclude_tile). Test on a
    # single held-out tile (test_tile). Existing configs (no tile_size) are unaffected.
    if dcfg.get('tile_size'):
        ts   = dcfg['tile_size']
        grid = [(y, x) for y in range(0, 800, ts) for x in range(0, 800, ts)]
        if dcfg.get('train_include_tiles'):            # explicit list of training tiles (y, x origins)
            allowed = [tuple(t) for t in dcfg['train_include_tiles']]
            excl_origin = None
        else:                                          # legacy: all tiles minus one held-out
            excl = tuple(dcfg['train_exclude_tile']) if dcfg.get('train_exclude_tile') else None
            excl_origin = (excl[0], excl[2]) if excl else None
            allowed = [g for g in grid if g != excl_origin]
        train_ds, val_ds, _ = split_fn(**split_kw)                 # full-domain train (roi=None)
        train_ds.fit_normalisation()
        stats = train_ds.get_stats()
        if z_range is not None:
            stats['z_range'] = z_range
        train_ds.patch_size    = ts
        train_ds.allowed_tiles = allowed
        train_ds.augment       = dcfg.get('augment', False)
        if dcfg.get('free_crop') and excl is not None:
            # arbitrary crop origins (not the fixed 4x4 grid), rejecting any
            # window that overlaps the held-out tile
            train_ds.allowed_tiles  = None
            train_ds.exclude_region = excl
        _, _, test_ds = split_fn(**{**split_kw, 'roi': tuple(dcfg['test_tile'])})
        val_ds._stats = test_ds._stats = stats
        print(f'Tile-holdout: train {len(allowed)} tiles (exclude {excl_origin}); '
              f'augment={train_ds.augment} free_crop={bool(dcfg.get("free_crop"))}; '
              f'test_tile={dcfg["test_tile"]}; train N={len(train_ds)} test N={len(test_ds)}')
        return train_ds, val_ds, test_ds, stats

    train_ds, val_ds, test_ds = split_fn(**split_kw)
    train_ds.fit_normalisation()
    stats = train_ds.get_stats()
    if z_range is not None:
        stats['z_range'] = z_range
    val_ds._stats = test_ds._stats = stats
    # tile training: random crops for TRAIN only; test stays full-domain
    if dcfg.get('patch_size'):
        train_ds.patch_size = dcfg['patch_size']
    return train_ds, val_ds, test_ds, stats


def build_loaders(train_ds, val_ds, test_ds, cfg: dict):
    dcfg = cfg['data']
    nw   = dcfg.get('num_workers', 2)
    # val/test may be full-domain even when training on patches -> force batch 1
    eval_bs = 1 if dcfg.get('patch_size') else dcfg['batch_size']
    return (
        DataLoader(train_ds, shuffle=True,  batch_size=dcfg['batch_size'], num_workers=nw, pin_memory=True),
        DataLoader(val_ds,   shuffle=False, batch_size=eval_bs,            num_workers=nw, pin_memory=True),
        DataLoader(test_ds,  shuffle=False, batch_size=eval_bs,            num_workers=nw, pin_memory=True),
    )


# ── Model / loss ──────────────────────────────────────────────────────────────

def build_model(cfg: dict):
    mc      = cfg['model']
    z_range = cfg['data'].get('z_range', None)
    n_z_eff = (z_range[1] - z_range[0]) if z_range else N_Z
    z_start = z_range[0] if z_range else 0
    name    = mc.get('name', 'UNet3DFiLM')
    if name != 'UNet3DFiLM':
        raise ValueError(f'Unsupported model name: {name!r} (only UNet3DFiLM is supported)')
    return UNet3DFiLM(
        in_ch=mc.get('in_ch', 12),
        spatial_ch=mc.get('spatial_ch', 4),
        base_ch=mc.get('base_ch', 32),
        dropout=mc.get('dropout', 0.0),
        n_z=n_z_eff,
        z_start=z_start,
        use_attention=mc.get('use_attention', False),
        attn_heads=mc.get('attn_heads', 4),
        n_down=mc.get('n_down', 3),
    )


def build_loss(cfg: dict, stats: dict = None) -> PhysicsLoss:
    lc = cfg['loss']
    pm_mean = pm_std = None
    if stats is not None:
        zr      = cfg['data'].get('z_range', None)
        pm_mean = stats['pm25']['mean'][zr[0]:zr[1]] if zr else stats['pm25']['mean']
        pm_std  = stats['pm25']['std'][zr[0]:zr[1]]  if zr else stats['pm25']['std']
    return PhysicsLoss(
        lambda_nonneg=lc.get('lambda_nonneg', 0.1),
        lambda_anom=lc.get('lambda_anom', 0.0),
        lambda_grad=lc.get('lambda_grad', 0.0),
        lambda_pattern=lc.get('lambda_pattern', 0.0),
        pattern_min_std=lc.get('pattern_min_std', 0.01),
        lambda_spectral=lc.get('lambda_spectral', 0.0),
        high_weight=lc.get('high_weight', 0.0),
        pm_mean=pm_mean, pm_std=pm_std,
    )


# ── Optimizer / scheduler ─────────────────────────────────────────────────────

def build_optimizer(cfg: dict, model: torch.nn.Module):
    tc = cfg['training']
    return torch.optim.AdamW(
        model.parameters(), lr=tc['lr'],
        weight_decay=tc.get('weight_decay', 1e-4),
    )


def build_scheduler(cfg: dict, optimizer):
    sc   = cfg['training'].get('scheduler', {})
    name = sc.get('name', 'plateau')
    if name == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min',
            patience=sc.get('patience', 10),
            factor=sc.get('factor', 0.5),
            min_lr=sc.get('min_lr', 1e-6),
        )
    if name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg['training']['epochs'],
            eta_min=sc.get('min_lr', 1e-6),
        )
    raise ValueError(f'Unknown scheduler: {name}')


# ── Denormalise ───────────────────────────────────────────────────────────────

def _denorm(x: torch.Tensor, stats: dict) -> torch.Tensor:
    """Reverse per-level z-score + log1p. x: (B, N_Z, H, W)."""
    zr   = stats.get('z_range', None)
    mean = stats['pm25']['mean'][zr[0]:zr[1]] if zr else stats['pm25']['mean']
    std  = stats['pm25']['std'][zr[0]:zr[1]]  if zr else stats['pm25']['std']
    mean = torch.tensor(mean, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    std  = torch.tensor(std,  dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    return torch.expm1(x * std + mean)


# ── Train / eval ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, loss_fn, optimizer, cfg, device, stats):
    model.train()
    grad_clip = cfg['training'].get('grad_clip', 1.0)
    log_every = cfg['logging'].get('log_every', 0)
    sums = {}
    sse, cnt, n = 0.0, 0, 0

    for i, (x, y, mask) in enumerate(loader):
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        pred = model(x)
        loss, bd = loss_fn(pred, y, mask)

        if not loss.isfinite():
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        for k, v in bd.items():
            sums[k] = sums.get(k, 0.0) + v
        with torch.no_grad():
            diff = _denorm(pred, stats)[mask] - _denorm(y, stats)[mask]
            sse += (diff ** 2).sum().item()
            cnt += mask.sum().item()
        n += 1

        if log_every and (i + 1) % log_every == 0:
            print(f'  batch {i+1}/{len(loader)}  loss={bd["total"]:.4f}')

    metrics = {k: v / n for k, v in sums.items()} if n > 0 else {}
    metrics.setdefault('total', float('inf'))   # guard: epoch with no finite batch
    metrics['rmse'] = math.sqrt(sse / max(cnt, 1))
    return metrics


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, stats, low_level_idx: int = 8):
    model.eval()
    sums = {}
    sse = sse_low = sse_high = 0.0
    ae = cnt = cnt_low = cnt_high = 0
    n = 0

    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        pred = model(x)
        _, bd = loss_fn(pred, y, mask)
        for k, v in bd.items():
            sums[k] = sums.get(k, 0.0) + v

        pp = _denorm(pred, stats)
        tp = _denorm(y,    stats)
        diff = pp[mask] - tp[mask]
        sse += (diff ** 2).sum().item()
        ae  += diff.abs().sum().item()
        cnt += mask.sum().item()

        m_low  = mask[:, :low_level_idx]
        m_high = mask[:, low_level_idx:]
        sse_low  += ((pp[:, :low_level_idx][m_low]  - tp[:, :low_level_idx][m_low])  ** 2).sum().item()
        sse_high += ((pp[:, low_level_idx:][m_high] - tp[:, low_level_idx:][m_high]) ** 2).sum().item()
        cnt_low  += m_low.sum().item()
        cnt_high += m_high.sum().item()
        n += 1

    metrics = {k: v / n for k, v in sums.items()}
    metrics['rmse']      = math.sqrt(sse      / max(cnt,      1))
    metrics['rmse_low']  = math.sqrt(sse_low  / max(cnt_low,  1))
    metrics['rmse_high'] = math.sqrt(sse_high / max(cnt_high, 1))
    metrics['mae']       = ae / max(cnt, 1)
    return metrics


@torch.no_grad()
def evaluate_r2_by_level(model, loader, device, stats, n_z=N_Z):
    model.eval()
    ss_res   = np.zeros(n_z)
    sum_y    = np.zeros(n_z)
    sum_y_sq = np.zeros(n_z)
    count    = np.zeros(n_z, dtype=np.int64)

    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        pp = _denorm(model(x), stats).cpu().numpy()
        tp = _denorm(y,        stats).cpu().numpy()
        mn = mask.cpu().numpy()

        for lev in range(n_z):
            m = mn[:, lev]
            if not m.any():
                continue
            p, t = pp[:, lev][m], tp[:, lev][m]
            ss_res[lev]   += ((p - t) ** 2).sum()
            sum_y[lev]    += t.sum()
            sum_y_sq[lev] += (t ** 2).sum()
            count[lev]    += t.size

    ss_tot = np.zeros(n_z)
    for lev in range(n_z):
        if count[lev] > 0:
            mean_y = sum_y[lev] / count[lev]
            ss_tot[lev] = sum_y_sq[lev] - count[lev] * mean_y ** 2
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


# ── Checkpointing ─────────────────────────────────────────────────────────────

def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val, cfg, stats):
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss':        best_val,
        'config':               cfg,
        'norm_stats':           stats,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler and ckpt.get('scheduler_state_dict'):
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    return ckpt['epoch'], ckpt['best_val_loss'], ckpt.get('norm_stats')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg['experiment'].get('seed', 42))

    # Fixed input shapes (full-domain, batch 1) -> let cudnn autotune the conv
    # algorithm per shape. Without this the heuristic can pick a much slower 3D
    # conv kernel for some z-depths (e.g. z=16 ran ~2x slower than z=15).
    torch.backends.cudnn.benchmark = True

    out_dir  = Path(cfg['experiment']['output_dir']) / cfg['experiment']['name']
    ckpt_dir = out_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out_dir / 'config.yaml')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print('Building datasets...')
    train_ds, val_ds, test_ds, stats = build_datasets(cfg)
    train_loader, val_loader, test_loader = build_loaders(train_ds, val_ds, test_ds, cfg)

    model    = build_model(cfg).to(device)
    if cfg['model'].get('zero_height', False):        # no-height ablation: blank the
        model.height_enc.zero_()                       # per-level height encoding (persistent
        print('height encoding zeroed (no-height ablation)')  # buffer -> saved in checkpoint)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model: UNet3DFiLM  base_ch={cfg["model"].get("base_ch", 32)}  params: {n_params:,}')

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    loss_fn   = build_loss(cfg, stats).to(device)

    start_epoch, best_val = 0, float('inf')
    if args.resume:
        start_epoch, best_val, _ = load_checkpoint(args.resume, model, optimizer, scheduler)
        print(f'Resumed from epoch {start_epoch}  best_val={best_val:.6f}')

    has_val     = len(val_ds) > 0
    tc          = cfg['training']
    n_epochs    = tc['epochs']
    patience    = tc.get('early_stop_patience', 0)
    no_improve  = 0
    use_plateau = tc.get('scheduler', {}).get('name', 'plateau') == 'plateau'

    logger = CSVLogger(out_dir / 'metrics.csv')
    writer = SummaryWriter(log_dir=str(out_dir / 'tb'))

    if not has_val:
        print('No val set — using train loss for scheduler and checkpoint selection.')

    print(f'\nTraining for {n_epochs} epochs...')
    for epoch in range(start_epoch, n_epochs):
        t0      = time.time()
        train_m = train_one_epoch(model, train_loader, loss_fn, optimizer, cfg, device, stats)

        if has_val:
            val_m   = evaluate(model, val_loader, loss_fn, device, stats)
            monitor = val_m['total']
        else:
            val_m   = {}
            monitor = train_m['total']

        if use_plateau:
            scheduler.step(monitor)
        else:
            scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        logger.log({'epoch': epoch + 1, 'lr': lr,
                    **{f'train_{k}': v for k, v in train_m.items()},
                    **{f'val_{k}':   v for k, v in val_m.items()}})

        writer.add_scalar('loss/train', train_m['total'], epoch + 1)
        writer.add_scalar('rmse/train', train_m['rmse'],  epoch + 1)
        if has_val:
            writer.add_scalar('loss/val',      val_m['total'],    epoch + 1)
            writer.add_scalar('rmse/val',      val_m['rmse'],     epoch + 1)
            writer.add_scalar('rmse_low/val',  val_m['rmse_low'], epoch + 1)
            writer.add_scalar('rmse_high/val', val_m['rmse_high'],epoch + 1)
        writer.add_scalar('lr', lr, epoch + 1)

        val_str = (f'  val={val_m["total"]:.3e}  rmse_val={val_m["rmse"]:.4f}'
                   if has_val else '')
        print(f'Epoch {epoch+1:3d}/{n_epochs}  '
              f'train={train_m["total"]:.3e}{val_str}  '
              f'rmse={train_m["rmse"]:.4f}  lr={lr:.1e}  {time.time()-t0:.0f}s')

        save_checkpoint(ckpt_dir / 'last.pt', model, optimizer, scheduler,
                        epoch + 1, best_val, cfg, stats)
        if monitor < best_val:
            best_val   = monitor
            no_improve = 0
            save_checkpoint(ckpt_dir / 'best.pt', model, optimizer, scheduler,
                            epoch + 1, best_val, cfg, stats)
            print(f'  -> best  monitor={best_val:.3e}')
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                print(f'Early stop at epoch {epoch + 1}')
                break

    logger.close()
    writer.close()

    # ── Test evaluation ───────────────────────────────────────────────────────
    print('\nTest evaluation (best checkpoint)...')
    load_checkpoint(ckpt_dir / 'best.pt', model, optimizer, scheduler)
    test_m = evaluate(model, test_loader, loss_fn, device, stats)
    print('Test (RMSE / MAE in μg/m³):')
    for k, v in test_m.items():
        print(f'  {k}: {v:.6f}')

    z_range_cfg = cfg['data'].get('z_range', None)
    n_z_eff     = (z_range_cfg[1] - z_range_cfg[0]) if z_range_cfg else N_Z
    z_start     = z_range_cfg[0] if z_range_cfg else 0

    r2 = evaluate_r2_by_level(model, test_loader, device, stats, n_z=n_z_eff)
    z_heights = test_ds.z_levels
    print(f'\n{"z":>3}  {"Height(m)":>9}  {"R²":>8}')
    print('-' * 26)
    for lev in range(n_z_eff):
        abs_lev = z_start + lev
        marker  = '  <- ~0-100m' if abs_lev < 8 else ''
        print(f'{abs_lev:>3}  {z_heights[abs_lev]:>9.1f}  {r2[lev]:>8.4f}{marker}')

    test_m['r2_by_level'] = r2.tolist()
    with open(out_dir / 'test_results.json', 'w') as f:
        json.dump(test_m, f, indent=2)


if __name__ == '__main__':
    main()
