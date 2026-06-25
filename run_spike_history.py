"""
run_spike_history.py - Compare RNN models with vs. without spike history input.

Motivation:
  Izhikevich neurons are deterministic and non-Poisson, so models that see only
  the stimulus I(t) cannot learn temporal spike dependencies (refractory period,
  burst structure, adaptation). Adding k(t-1) as a second input feature gives
  the model direct access to its own recent spiking history, letting it learn
  those patterns through backprop.

  NOTE: evaluation uses teacher-forcing — the TRUE k(t-1) from y_test is fed
  as input at test time.  This is optimistic for the history model but
  consistent with the existing evaluate_trainer framework, and allows an
  upper-bound comparison.

Input feature layout:
  Baseline:       X[:, :, 0] = I(t)                 (input_size=1)
  + Spike history: X[:, :, 0] = I(t)                 (input_size=2)
                   X[:, :, 1] = k(t-1)

Usage:
    python run_spike_history.py               # cell types 1 2 3 4
    python run_spike_history.py 3             # Tonic bursting
    python run_spike_history.py 1 3 4         # Multiple cell types
    python run_spike_history.py all           # All available cell types
    python run_spike_history.py 3 --models VanillaRNN LearnableTau
    python run_spike_history.py 3 --n_epochs 100 --hidden_size 32
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

torch.set_num_threads(1)

from models import build_model, MODEL_REGISTRY
from data import generate_training_data, generate_test_trial
from train import Trainer
from evaluate import evaluate_trainer
from izhikevich_configs import index_to_name


# =============================================================================
# Configuration
# =============================================================================

N_TRIALS      = 500
N_EPOCHS      = 200
LR            = 1e-3
BATCH_SIZE    = 64
BIN_SIZE_MS   = 1.0
HIDDEN_SIZE   = 64
N_TEST_TRIALS = 5
SEED          = 42

AVAILABLE_TYPES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21]

DEFAULT_MODELS = ['VanillaRNN', 'LearnableTau', 'FixedSpectrum', 'Partitioned', 'PLRNN']

PLOT_ROOT = Path(__file__).parent / 'plots' / 'spike_history'


# =============================================================================
# Data augmentation
# =============================================================================

def augment_with_spike_history(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Append k(t-1) as a second input channel alongside I(t).

    Args:
        X: (n, seq_len, 1)  — stimulus current I(t)
        y: (n, seq_len)     — spike counts k(t)

    Returns:
        X_aug: (n, seq_len, 2)  — channel 0: I(t), channel 1: k(t-1)
    """
    n, T, _ = X.shape
    X_aug = np.zeros((n, T, 2), dtype=np.float32)
    X_aug[:, :, 0] = X[:, :, 0]    # I(t)
    X_aug[:, 1:, 1] = y[:, :-1]    # k(t-1); k(-1) = 0 for t=0
    return X_aug


# =============================================================================
# Training
# =============================================================================

def _train(model_name: str, input_size: int,
           X: np.ndarray, y: np.ndarray, lengths: np.ndarray,
           hidden_size: int, n_epochs: int, seed: int) -> Trainer:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(model_name, input_size=input_size,
                        hidden_size=hidden_size, dt=BIN_SIZE_MS, activation='tanh')
    trainer = Trainer(model, lr=LR, batch_size=BATCH_SIZE, device='cpu')
    trainer.fit(X, y, n_epochs=n_epochs, lengths=lengths,
                verbose=False, use_freeze_schedule=True)
    return trainer


# =============================================================================
# Plotting
# =============================================================================

def _get_isis(k: np.ndarray) -> np.ndarray:
    """Return flat ISI array from a 1-D spike count array."""
    times = np.repeat(np.arange(len(k)), k.astype(int))
    return np.diff(times).astype(float) if len(times) >= 2 else np.array([])


def plot_metric_comparison(baseline_metrics: dict, history_metrics: dict,
                           model_names: list, cell_type: int, out_dir: Path):
    """
    Bar chart showing baseline vs. spike-history for three key metrics.
    """
    ct_name = index_to_name.get(cell_type, f'Type {cell_type}')
    x = np.arange(len(model_names))
    w = 0.35

    metrics_cfg = [
        ('isi_wasserstein', 'ISI Wasserstein ↓'),
        ('bps',             'Bits per Spike ↑'),
        ('pearson_r',       'Pearson r ↑'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)

    for ax, (key, label) in zip(axes, metrics_cfg):
        base_vals = [baseline_metrics[m].get(key, np.nan) for m in model_names]
        hist_vals = [history_metrics[m].get(key, np.nan) for m in model_names]

        bars_b = ax.bar(x - w / 2, base_vals, w,
                        label='Baseline', color='steelblue', alpha=0.85)
        bars_h = ax.bar(x + w / 2, hist_vals, w,
                        label='+ Spike history k(t-1)', color='tomato', alpha=0.85)

        for bar in list(bars_b) + list(bars_h):
            v = bar.get_height()
            if np.isfinite(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v,
                        f'{v:.2f}', ha='center', va='bottom', fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=25, ha='right', fontsize=9)
        ax.set_title(f'{label}')
        ax.legend(fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle(f'Baseline vs. spike history — CT {cell_type}: {ct_name}',
                 fontsize=12, fontweight='bold')
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'ct{cell_type}_metric_comparison.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_isi_distributions(trainer_base: Trainer, trainer_hist: Trainer,
                           test_trials: list, model_name: str,
                           cell_type: int, out_dir: Path,
                           n_poisson_samples: int = 10):
    """
    Three-panel ISI histogram: ground truth | baseline | + spike history.
    Poisson samples from predicted λ to generate simulated spike trains.
    """
    ct_name = index_to_name.get(cell_type, f'Type {cell_type}')

    gt_isis, base_isis, hist_isis = [], [], []

    for I_b, y_b in test_trials:
        X1 = I_b[np.newaxis, :, np.newaxis].astype(np.float32)
        X2 = augment_with_spike_history(X1, y_b[np.newaxis])

        lam_base = trainer_base.predict(X1)[0]
        lam_hist = trainer_hist.predict(X2)[0]

        gt_isis.append(_get_isis(y_b))
        for _ in range(n_poisson_samples):
            bi = _get_isis(np.random.poisson(lam_base))
            hi = _get_isis(np.random.poisson(lam_hist))
            if len(bi): base_isis.append(bi)
            if len(hi): hist_isis.append(hi)

    gt_all   = np.concatenate(gt_isis)   if gt_isis   else np.array([np.nan])
    base_all = np.concatenate(base_isis) if base_isis else np.array([np.nan])
    hist_all = np.concatenate(hist_isis) if hist_isis else np.array([np.nan])

    finite_gt = gt_all[np.isfinite(gt_all)]
    max_isi = float(np.percentile(finite_gt, 99) * 1.5) if len(finite_gt) > 1 else 200.0
    bins = np.linspace(0, max_isi, 60)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), constrained_layout=True)
    panels = [
        (gt_all,   'Ground truth',           'k'),
        (base_all, 'Baseline (no history)',   'steelblue'),
        (hist_all, '+ Spike history k(t-1)', 'tomato'),
    ]
    for ax, (arr, title, color) in zip(axes, panels):
        finite = arr[np.isfinite(arr)]
        if len(finite) > 1:
            ax.hist(finite, bins=bins, color=color, alpha=0.75,
                    density=True, edgecolor='none')
            mean_isi = finite.mean()
            ax.axvline(mean_isi, color='k', ls='--', lw=1.2,
                       label=f'mean={mean_isi:.1f}')
            ax.legend(fontsize=8)
        ax.set_xlabel('ISI (bins / ms)')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle(f'ISI distributions — {model_name} — CT {cell_type}: {ct_name}',
                 fontsize=11, fontweight='bold')
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'ct{cell_type}_{model_name}_isi.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_rate_traces(trainer_base: Trainer, trainer_hist: Trainer,
                     I_b: np.ndarray, y_b: np.ndarray,
                     model_name: str, cell_type: int, out_dir: Path):
    """
    Single-trial rate trace comparison: GT spike / baseline λ / history λ.
    """
    ct_name = index_to_name.get(cell_type, f'Type {cell_type}')
    from scipy.ndimage import gaussian_filter1d

    X1 = I_b[np.newaxis, :, np.newaxis].astype(np.float32)
    X2 = augment_with_spike_history(X1, y_b[np.newaxis])

    lam_base = trainer_base.predict(X1)[0]
    lam_hist = trainer_hist.predict(X2)[0]
    gt_smooth = gaussian_filter1d(y_b.astype(float), sigma=15)

    t = np.arange(len(I_b))
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True, constrained_layout=True)

    axes[0].fill_between(t, I_b, alpha=0.6, color='gray')
    axes[0].set_ylabel('Current I(t)')

    axes[1].plot(t, gt_smooth, 'k', lw=1.2, label='GT (smoothed)')
    axes[1].plot(t, lam_base, color='steelblue', lw=1.0, alpha=0.85, label='Baseline λ(t)')
    axes[1].plot(t, lam_hist, color='tomato',    lw=1.0, alpha=0.85, label='+ History λ(t)')
    axes[1].set_ylabel('Firing rate (sp/bin)')
    axes[1].legend(fontsize=8)

    axes[2].plot(t, np.abs(gt_smooth - lam_base), color='steelblue', lw=0.8, label='|error| baseline')
    axes[2].plot(t, np.abs(gt_smooth - lam_hist), color='tomato',    lw=0.8, label='|error| history')
    axes[2].set_ylabel('|Error|')
    axes[2].set_xlabel('Time (bins / ms)')
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle(f'Rate traces — {model_name} — CT {cell_type}: {ct_name}',
                 fontsize=11, fontweight='bold')
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'ct{cell_type}_{model_name}_rate_trace.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compare RNN models with and without spike history input.'
    )
    parser.add_argument('cell_types', nargs='*', default=['1', '2', '3', '4'],
                        help='Cell type indices or "all"')
    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS,
                        help=f'Models to include (default: {DEFAULT_MODELS})')
    parser.add_argument('--hidden_size', type=int, default=HIDDEN_SIZE)
    parser.add_argument('--n_epochs', type=int, default=N_EPOCHS)
    parser.add_argument('--n_trials', type=int, default=N_TRIALS)
    args = parser.parse_args()

    # Parse cell types
    if len(args.cell_types) == 1 and args.cell_types[0].lower() == 'all':
        valid_types = AVAILABLE_TYPES
    else:
        valid_types = [int(x) for x in args.cell_types if int(x) in AVAILABLE_TYPES]
    if not valid_types:
        print("No valid cell types.")
        return

    model_names = [m for m in args.models if m in MODEL_REGISTRY]
    if not model_names:
        print(f"No valid models. Choose from: {list(MODEL_REGISTRY.keys())}")
        return

    print(f"Cell types:  {valid_types}")
    print(f"Models:      {model_names}")
    print(f"Epochs:      {args.n_epochs}  |  Trials: {args.n_trials}  |  Hidden: {args.hidden_size}\n")

    for ct in valid_types:
        ct_name = index_to_name.get(ct, f'Type {ct}')
        print(f"\n{'='*60}")
        print(f"  Cell type {ct}: {ct_name}")
        print(f"{'='*60}")

        # Generate data (train / test split)
        X, y, lengths = generate_training_data(
            ct, n_trials=args.n_trials, bin_size_ms=BIN_SIZE_MS,
            seed=SEED, verbose=True,
        )
        X_tr,  y_tr,  l_tr  = X[:-N_TEST_TRIALS], y[:-N_TEST_TRIALS], lengths[:-N_TEST_TRIALS]
        X_te,  y_te,  l_te  = X[-N_TEST_TRIALS:],  y[-N_TEST_TRIALS:],  lengths[-N_TEST_TRIALS:]

        # Augmented versions (teacher-forcing: true k(t-1) appended)
        X_tr_aug = augment_with_spike_history(X_tr, y_tr)
        X_te_aug = augment_with_spike_history(X_te, y_te)

        out_dir = PLOT_ROOT / f'ct{ct}_{ct_name.lower().replace(" ", "_")}'

        baseline_metrics = {}
        history_metrics  = {}

        # One clean trial for rate-trace plots
        I_clean, y_clean = generate_test_trial(ct, bin_size_ms=BIN_SIZE_MS)
        test_trials = [generate_test_trial(ct, bin_size_ms=BIN_SIZE_MS)
                       for _ in range(N_TEST_TRIALS)]

        for mname in model_names:
            print(f"\n  [{mname}]")

            # Baseline: only I(t)
            print(f"    Training baseline (input_size=1)...")
            t_base = _train(mname, input_size=1,
                            X=X_tr, y=y_tr, lengths=l_tr,
                            hidden_size=args.hidden_size,
                            n_epochs=args.n_epochs, seed=SEED)
            m_base = evaluate_trainer(t_base, X_te, y_te, l_te)
            baseline_metrics[mname] = m_base

            # With spike history: I(t) + k(t-1)
            print(f"    Training + spike history (input_size=2)...")
            t_hist = _train(mname, input_size=2,
                            X=X_tr_aug, y=y_tr, lengths=l_tr,
                            hidden_size=args.hidden_size,
                            n_epochs=args.n_epochs, seed=SEED)
            m_hist = evaluate_trainer(t_hist, X_te_aug, y_te, l_te)
            history_metrics[mname] = m_hist

            # Console comparison table
            print(f"    {'Metric':<22} {'Baseline':>10} {'+ History':>10} {'Δ':>10}")
            print(f"    {'-'*54}")
            for key in ('bps', 'pearson_r', 'isi_wasserstein', 'mean_rate_err'):
                b = m_base.get(key, np.nan)
                h = m_hist.get(key, np.nan)
                d = h - b if np.isfinite(b) and np.isfinite(h) else np.nan
                arrow = ('↑' if d > 0 else '↓') if np.isfinite(d) else ''
                print(f"    {key:<22} {b:>10.4f} {h:>10.4f} {d:>+9.4f}{arrow}")

            # Per-model plots
            plot_isi_distributions(t_base, t_hist, test_trials, mname, ct, out_dir)
            plot_rate_traces(t_base, t_hist, I_clean, y_clean, mname, ct, out_dir)

        # Summary bar chart across all models for this cell type
        plot_metric_comparison(baseline_metrics, history_metrics,
                               model_names, ct, out_dir)

    print(f"\nDone. Plots saved to {PLOT_ROOT}/")


if __name__ == '__main__':
    main()
