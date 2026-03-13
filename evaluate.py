"""
evaluate.py - Model evaluation metrics for RNN spike prediction.

Primary metric: bits per spike (BPS)
    BPS = (NLL_baseline - NLL_model) / N_spikes / log(2)
    > 0: model captures temporal structure beyond mean rate
    = 0: equivalent to a homogeneous Poisson at mean rate
    < 0: worse than mean rate

Supporting metrics:
    test_nll      - Poisson NLL on held-out trials (lower = better)
    pearson_r     - correlation between predicted rate and smoothed GT rate
    mean_rate_err - |mean_pred - mean_gt| / mean_gt (fractional rate error)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Optional, Tuple

from models import MODEL_COLORS
from izhikevich_configs import index_to_name


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def _poisson_nll(y_pred: np.ndarray, y_true: np.ndarray,
                 lengths: Optional[np.ndarray] = None) -> float:
    """
    Mean Poisson NLL: E[lambda - k * log(lambda)]
    Accepts batched (n, T) or single (T,) arrays.
    If lengths provided, masks padded timesteps.
    """
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    if y_pred.ndim == 1:
        y_pred = y_pred[None]
        y_true = y_true[None]

    per_step = y_pred - y_true * np.log(y_pred + 1e-8)  # (n, T)

    if lengths is not None:
        mask = np.zeros_like(per_step, dtype=bool)
        for i, L in enumerate(lengths):
            mask[i, :L] = True
        return per_step[mask].mean()
    return per_step.mean()


def evaluate_trainer(trainer,
                     X_test: np.ndarray,
                     y_test: np.ndarray,
                     lengths_test: Optional[np.ndarray] = None,
                     smooth_sigma: float = 15.0) -> dict:
    """
    Compute evaluation metrics for a single trainer on held-out test data.

    Args:
        trainer:      trained Trainer instance
        X_test:       (n, T, 1) input array
        y_test:       (n, T) spike count targets
        lengths_test: (n,) actual lengths for masking (None = full sequences)
        smooth_sigma: Gaussian sigma (bins) for smoothing GT before Pearson r

    Returns:
        dict with keys: test_nll, bps, pearson_r, mean_rate_err
    """
    y_pred = trainer.predict(X_test)  # (n, T)

    # --- Test NLL ---
    test_nll = _poisson_nll(y_pred, y_test, lengths_test)

    # --- Baseline NLL (homogeneous Poisson at mean rate) ---
    if lengths_test is not None:
        mask = np.zeros(y_test.shape, dtype=bool)
        for i, L in enumerate(lengths_test):
            mask[i, :L] = True
        mean_rate = y_test[mask].mean()
    else:
        mean_rate = y_test.mean()
    baseline_nll = _poisson_nll(
        np.full_like(y_pred, mean_rate), y_test, lengths_test
    )

    # --- BPS ---
    n_spikes = y_test.sum()
    if n_spikes > 0:
        bps = (baseline_nll - test_nll) / (n_spikes / y_test.size) / np.log(2)
    else:
        bps = np.nan

    # --- Pearson r: per-trial, averaged across all test trials ---
    # Smoothing the sparse GT spike train makes it comparable to the predicted rate.
    # Computing per-trial and averaging removes the dependence on a single noisy trial.
    rs = []
    for i in range(len(y_pred)):
        L = int(lengths_test[i]) if lengths_test is not None else y_pred.shape[1]
        gt_smooth = gaussian_filter1d(y_test[i, :L].astype(float), sigma=smooth_sigma)
        pr_trial  = y_pred[i, :L]
        if gt_smooth.std() > 0 and pr_trial.std() > 0:
            rs.append(float(np.corrcoef(gt_smooth, pr_trial)[0, 1]))
    pearson_r = float(np.mean(rs)) if rs else np.nan

    # --- Mean rate error ---
    pred_mean = y_pred.mean() if lengths_test is None else y_pred[mask].mean()
    mean_rate_err = abs(pred_mean - mean_rate) / (mean_rate + 1e-8)

    return {
        'test_nll':      float(test_nll),
        'bps':           float(bps),
        'pearson_r':     float(pearson_r),
        'mean_rate_err': float(mean_rate_err),
    }


# ---------------------------------------------------------------------------
# Batch evaluation across all models / cell types / grid points
# ---------------------------------------------------------------------------

def evaluate_all(
    trainers_by_config: dict,
    test_data: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_test_trials: int = 20,
) -> pd.DataFrame:
    """
    Evaluate all trained models.

    Args:
        trainers_by_config: {(hs, tlr): {ct: {mname: Trainer}}}
        test_data:          {ct: (X, y, lengths)}  — training data reused as proxy;
                            evaluation uses held-out slices (last n_test_trials)
        n_test_trials:      how many trials to hold out for evaluation

    Returns:
        Tidy DataFrame with columns:
            hidden_size, tau_lr, cell_type, model, test_nll, bps, pearson_r, mean_rate_err
    """
    rows = []
    for (hs, tlr), by_ct in trainers_by_config.items():
        for ct, trainers in by_ct.items():
            X, y, lengths = test_data[ct]
            # Use last n_test_trials as a held-out evaluation set
            X_eval = X[-n_test_trials:]
            y_eval = y[-n_test_trials:]
            l_eval = lengths[-n_test_trials:]

            for mname, trainer in trainers.items():
                metrics = evaluate_trainer(trainer, X_eval, y_eval, l_eval)
                rows.append({
                    'hidden_size': hs,
                    'tau_lr':      tlr,
                    'cell_type':   ct,
                    'cell_name':   index_to_name.get(ct, f'Type {ct}'),
                    'model':       mname,
                    **metrics,
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary heatmap
# ---------------------------------------------------------------------------

def plot_metric_heatmap(df: pd.DataFrame, metric: str = 'bps',
                        plot_root: Path = None):
    """
    One heatmap per grid point (hidden_size × tau_lr):
      rows = models, cols = cell types, color = metric value.
    Saved to plot_root/metric_heatmap_{metric}.png
    """
    from plotting import PLOT_ROOT
    root = plot_root if plot_root is not None else PLOT_ROOT / 'tanh'

    grid_points = df[['hidden_size', 'tau_lr']].drop_duplicates().values
    models = df['model'].unique().tolist()
    cell_types = sorted(df['cell_type'].unique().tolist())
    cell_names = [index_to_name.get(ct, str(ct)) for ct in cell_types]

    n_grid = len(grid_points)
    ncols = min(n_grid, 4)
    nrows = (n_grid + ncols - 1) // ncols

    metric_label = {
        'bps': 'Bits per spike',
        'test_nll': 'Test NLL',
        'pearson_r': 'Pearson r',
        'mean_rate_err': 'Mean rate error',
    }.get(metric, metric)

    # Scientific colour maps: diverging for signed metrics, sequential for error/NLL
    _CMAPS = {
        'bps':           'PRGn',       # purple–green diverging
        'pearson_r':     'PuOr',       # purple–orange diverging
        'test_nll':      'plasma_r',   # sequential, lower = better
        'mean_rate_err': 'YlOrRd',     # sequential, lower = better
    }
    cmap = _CMAPS.get(metric, 'viridis')

    # Shared colour scale: symmetric around 0 for diverging metrics
    vals = df[metric].dropna()
    if metric in ('bps', 'pearson_r'):
        absmax = max(abs(vals.quantile(0.05)), abs(vals.quantile(0.95)))
        vmin, vmax = -absmax, absmax
    else:
        vmin, vmax = vals.quantile(0.05), vals.quantile(0.95)

    cell_w = max(1.1, 6.5 / max(len(cell_types), 1))
    model_h = max(0.55, 4.5 / max(len(models), 1))
    panel_w = len(cell_types) * cell_w + 1.2   # +1.2 for colorbar
    panel_h = len(models) * model_h + 0.8       # +0.8 for x-labels
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel_w * ncols, panel_h * nrows),
                             squeeze=False,
                             constrained_layout=True)

    for idx, (hs, tlr) in enumerate(grid_points):
        ax = axes[idx // ncols][idx % ncols]
        sub = df[(df['hidden_size'] == hs) & (df['tau_lr'] == tlr)]

        matrix = np.full((len(models), len(cell_types)), np.nan)
        for r, mname in enumerate(models):
            for c, ct in enumerate(cell_types):
                val = sub[(sub['model'] == mname) & (sub['cell_type'] == ct)][metric]
                if len(val):
                    matrix[r, c] = val.values[0]

        im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(cell_types)))
        ax.set_xticklabels(cell_names, rotation=35, ha='right', fontsize=8)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=8)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Annotate cells with contrasting text
        for r in range(len(models)):
            for c in range(len(cell_types)):
                v = matrix[r, c]
                if not np.isnan(v):
                    # Pick white or dark text for contrast
                    norm_v = (v - vmin) / (vmax - vmin + 1e-12)
                    txt_color = 'white' if (norm_v < 0.25 or norm_v > 0.75) else '#1a1a2e'
                    ax.text(c, r, f'{v:.2f}', ha='center', va='center',
                            fontsize=7, color=txt_color, fontweight='bold')

        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(metric_label, fontsize=8)
        cb.ax.tick_params(labelsize=7)
        cb.outline.set_visible(False)

    # Hide unused panels
    for idx in range(n_grid, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(metric_label, fontsize=13, fontweight='bold', y=1.01)
    out = root / f'metric_heatmap_{metric}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close(fig)
