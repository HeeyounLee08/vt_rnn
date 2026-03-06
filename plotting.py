"""
plotting.py - Visualization utilities for comparing RNN model performance.

Generates:
1. Training data validation plots (stimulus + spike rasters)
2. Loss curve comparison across models
3. Test-set prediction overlays (true vs predicted firing rate)
4. Tau distributions for CTRNN variants
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from pathlib import Path

import torch
from models import MODEL_COLORS
from izhikevich_configs import index_to_name

PLOT_ROOT = Path(__file__).parent / 'plots'
PLOT_ROOT.mkdir(exist_ok=True)


def get_plot_dir(cell_type: int, plot_root: Path = None) -> Path:
    """Return (and create) the per-cell-type subdirectory under plot_root."""
    root = plot_root if plot_root is not None else PLOT_ROOT
    name = index_to_name.get(cell_type, f'type_{cell_type}').lower().replace(' ', '_')
    d = root / f'ct{cell_type}_{name}'
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(fig, path: Path):
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close(fig)


def plot_training_data(X: np.ndarray, y: np.ndarray, lengths: np.ndarray,
                       cell_type: int, n_examples: int = 5, plot_root: Path = None):
    """Show a few example trials: stimulus (left) and spike counts (right)."""
    fig, axes = plt.subplots(n_examples, 2, figsize=(12, 2.5 * n_examples))
    if n_examples == 1:
        axes = axes.reshape(1, -1)

    idxs = np.random.choice(len(X), n_examples, replace=False)

    for i, idx in enumerate(idxs):
        n = int(lengths[idx])
        t = np.arange(n)
        axes[i, 0].plot(t, X[idx, :n, 0], 'b', lw=1.2)
        axes[i, 0].set_ylabel('Current')
        axes[i, 0].grid(alpha=0.3)
        axes[i, 1].bar(t, y[idx, :n], width=1.0, color='k', alpha=0.7)
        axes[i, 1].set_ylabel('Spikes')
        axes[i, 1].grid(alpha=0.3)
        axes[i, 1].text(0.98, 0.9, f'{int(y[idx].sum())} spk',
                        transform=axes[i, 1].transAxes, ha='right', fontsize=9)

    axes[0, 0].set_title('Stimulus')
    axes[0, 1].set_title('Spike Counts')
    axes[-1, 0].set_xlabel('Time bin (ms)')
    axes[-1, 1].set_xlabel('Time bin (ms)')
    fig.suptitle(f'Training Data - Cell Type {cell_type}', fontweight='bold')
    plt.tight_layout()
    _save(fig, get_plot_dir(cell_type, plot_root) / 'training_data.png')


def plot_loss_curves(trainers: Dict, cell_type: int, plot_root: Path = None):
    """Compare training loss curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, trainer in trainers.items():
        ax.plot(np.arange(1, len(trainer.train_losses) + 1), trainer.train_losses,
                color=MODEL_COLORS.get(name, 'gray'), lw=2,
                marker='o', markersize=3, label=name, alpha=0.85)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Poisson NLL Loss')
    ax.set_title(f'Training Loss - Cell Type {cell_type}')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, get_plot_dir(cell_type, plot_root) / 'loss_curves.png')


def plot_test_predictions(trainers: Dict, test_trials: list, cell_type: int, plot_root: Path = None):
    """
    For each test trial (column): stimulus+GT (top row) then per-model predictions.
    test_trials: list of (I_binned, y_binned) tuples
    """
    n_models = len(trainers)
    n_trials = len(test_trials)
    n_rows = n_models + 1

    fig, axes = plt.subplots(n_rows, n_trials,
                             figsize=(5 * n_trials, 2.5 * n_rows),
                             squeeze=False)

    for col, (I_binned, y_binned) in enumerate(test_trials):
        t = np.arange(len(I_binned))
        X_test = I_binned.reshape(1, -1, 1)
        gt_times = np.where(y_binned > 0)[0]

        # Top row: stimulus + GT spikes
        ax0 = axes[0, col]
        ax0_twin = ax0.twinx()
        ax0.plot(t, I_binned, 'b', lw=1, alpha=0.6)
        ax0_twin.vlines(gt_times, 0, 1, color='k', lw=1.2)
        ax0_twin.set_ylim(0, 3)
        ax0_twin.set_yticks([])
        if col == 0:
            ax0.set_ylabel('Current', color='b')
        ax0.set_title(f'Trial {col + 1}', fontsize=9)
        ax0.grid(alpha=0.2)

        for i, (name, trainer) in enumerate(trainers.items()):
            ax = axes[i + 1, col]
            color = MODEL_COLORS.get(name, 'gray')

            y_pred = trainer.predict(X_test)[0]
            y_sampled = np.random.poisson(y_pred)

            ax.vlines(gt_times, 1.1, 1.9, color='k', lw=1.2)
            ax.vlines(np.where(y_sampled > 0)[0], 0.1, 0.9,
                      color=color, lw=1.2, alpha=0.8)

            ax_r = ax.twinx()
            ax_r.plot(t, y_pred, color=color, lw=1.2, alpha=0.4, linestyle='--')
            ax_r.set_ylim(bottom=0)
            if col == n_trials - 1:
                ax_r.set_ylabel('Rate', fontsize=8)
            else:
                ax_r.set_yticks([])

            ax.set_ylim(0, 2)
            ax.set_yticks([0.5, 1.5])
            if col == 0:
                ax.set_yticklabels(['Pred', 'GT'], fontsize=8)
                ax.set_ylabel(name, fontsize=9)
            else:
                ax.set_yticklabels([])
            ax.grid(alpha=0.2)

        axes[-1, col].set_xlabel('Time (bins)')

    axes[0, 0].set_title(f'Cell Type {cell_type} — Trial 1', fontsize=9, fontweight='bold')
    fig.suptitle(f'Test Predictions - Cell Type {cell_type}', fontweight='bold', y=1.01)
    plt.tight_layout()
    _save(fig, get_plot_dir(cell_type, plot_root) / 'test_predictions.png')


def plot_tau_distributions(trainers: Dict, cell_type: int, plot_root: Path = None):
    """
    Visualize the timescale (tau) structure of each CTRNN variant.

    - LearnableTau:   histogram of learned taus (computed from trained rho)
    - FixedSpectrum:  histogram of fixed log-normal taus (from alpha buffer)
    - Partitioned:    two bars showing fast/slow compartment sizes
    - VanillaRNN:     no tau concept — shown as annotation only
    """
    fig, axes = plt.subplots(1, len(trainers), figsize=(4 * len(trainers), 4))
    fig.suptitle(f'Timescale (τ) Distributions - Cell Type {cell_type}', fontweight='bold')

    for ax, (name, trainer) in zip(axes, trainers.items()):
        color = MODEL_COLORS.get(name, 'gray')
        model = trainer.model
        ax.set_title(name, fontsize=10)

        if name == 'VanillaRNN':
            ax.text(0.5, 0.5, 'No τ\n(discrete RNN)', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_axis_off()

        elif name == 'Clockwork':
            periods = model.unit_periods.unique().cpu().numpy()
            units_per = model.hidden_size // len(periods)
            ax.bar([str(p) for p in periods], [units_per] * len(periods),
                   color=color, edgecolor='k', lw=0.8, alpha=0.8)
            ax.set_xlabel('Period (steps)')
            ax.set_ylabel('# units')
            ax.set_title(f'{name}\n[{len(periods)} modules, T-masked]', fontsize=9)

        elif name == 'LearnableTau':
            with torch.no_grad():
                alpha = model.get_alpha().cpu().numpy()
            taus = model.dt / alpha
            ax.hist(taus, bins=20, color=color, alpha=0.8, edgecolor='k', lw=0.5)
            ax.axvline(taus.mean(), color='k', ls='--', lw=1.2,
                       label=f'mean={taus.mean():.1f}ms')
            ax.set_xlabel('τ (ms)')
            ax.set_ylabel('# units')
            ax.legend(fontsize=8)
            ax.set_title(f'{name}\n[τ_min={model.tau_min}, τ_max={model.tau_max}]', fontsize=9)

        elif name == 'FixedSpectrum':
            alpha = model.alpha.cpu().numpy()
            taus = model.dt / alpha
            ax.hist(taus, bins=20, color=color, alpha=0.8, edgecolor='k', lw=0.5)
            ax.axvline(taus.mean(), color='k', ls='--', lw=1.2,
                       label=f'mean={taus.mean():.1f}ms')
            ax.set_xlabel('τ (ms)')
            ax.set_ylabel('# units')
            ax.legend(fontsize=8)
            ax.set_title(f'{name}\n[log-normal, fixed]', fontsize=9)

        elif name == 'Partitioned':
            alpha = model.alpha.cpu().numpy()
            taus = model.dt / alpha
            tau_fast = taus[:model.n_fast]
            tau_slow = taus[model.n_fast:]
            ax.bar(['Fast\n(τ={:.1f}ms)'.format(tau_fast[0]),
                    'Slow\n(τ={:.1f}ms)'.format(tau_slow[0])],
                   [model.n_fast, model.n_slow],
                   color=[color, 'lightcoral'], edgecolor='k', lw=0.8)
            ax.set_ylabel('# units')

        elif name == 'MultiPartitioned':
            labels = [f'τ={τ:.0f}ms' for τ in model.taus]
            cmap = plt.cm.plasma(np.linspace(0.1, 0.9, len(model.taus)))
            ax.bar(labels, model.sizes, color=cmap, edgecolor='k', lw=0.8)
            ax.set_ylabel('# units')
            ax.set_title(f'{name}\n[{len(model.taus)} compartments]', fontsize=9)

        elif name == 'LearnablePartitioned':
            with torch.no_grad():
                alpha = model.get_alpha().cpu().numpy()
            taus = model.dt / alpha
            tau_fast = taus[:model.n_fast]
            tau_slow = taus[model.n_fast:]
            ax.bar([f'Fast\n(τ={tau_fast[0]:.1f}ms)',
                    f'Slow\n(τ={tau_slow[0]:.1f}ms)'],
                   [model.n_fast, model.n_slow],
                   color=[color, 'wheat'], edgecolor='k', lw=0.8)
            ax.set_ylabel('# units')
            ax.set_title(f'{name}\n[learnable, 2 params]', fontsize=9)

        ax.grid(alpha=0.2)

    plt.tight_layout()
    _save(fig, get_plot_dir(cell_type, plot_root) / 'tau_distributions.png')


def plot_hidden_activations(trainer, I_binned: np.ndarray, y_binned: np.ndarray,
                            cell_type: int, model_name: str, plot_root: Path = None):
    """
    3-panel activation heatmap for a single model on a clean test trial.

    Units are sorted by:
    - VanillaRNN: activation variance (most active on top)
    - Clockwork:  period (fastest to slowest)
    - CTRNNs:     tau (fastest to slowest)
    """
    model = trainer.model
    X_norm = trainer._normalize(I_binned.reshape(1, -1, 1), fit=False)
    X_t = torch.FloatTensor(X_norm).to(trainer.device)

    model.eval()
    with torch.no_grad():
        y_pred, h_seq = model(X_t, return_sequence=True, return_hidden=True)
        y_pred = y_pred.squeeze(0).cpu().numpy()
        h_seq = h_seq.squeeze(0).cpu().numpy()  # (seq_len, hidden_size)

    act = np.abs(np.tanh(h_seq))  # (seq_len, hidden_size), values in [0, 1]

    # Sorting logic
    if model_name == 'VanillaRNN':
        sort_idx = np.argsort(act.var(axis=0))
        ylabel = 'Unit (sorted by variance)'
    elif model_name == 'Clockwork':
        sort_idx = np.argsort(model.unit_periods.cpu().numpy())
        ylabel = 'Unit (sorted by period)'
    else:
        if hasattr(model, 'get_alpha'):
            with torch.no_grad():
                alpha = model.get_alpha().cpu().numpy()
        else:
            alpha = model.alpha.cpu().numpy()
        taus = model.dt / alpha
        sort_idx = np.argsort(taus)
        ylabel = 'Unit (sorted by τ)'

    sorted_act = act[:, sort_idx]  # (seq_len, hidden_size)
    t = np.arange(len(I_binned))
    color = MODEL_COLORS.get(model_name, 'gray')

    fig, axes = plt.subplots(3, 1, figsize=(14, 6),
                             gridspec_kw={'height_ratios': [1, 3, 1]})

    # --- Top: stimulus ---
    axes[0].plot(t, I_binned, color='cornflowerblue', lw=1.2)
    axes[0].set_ylabel('Input', fontsize=8)
    axes[0].set_xlim(0, len(t) - 1)
    axes[0].set_xticks([])
    axes[0].grid(alpha=0.2)

    # --- Middle: heatmap ---
    im = axes[1].imshow(sorted_act.T, aspect='auto', origin='lower',
                        cmap='magma', vmin=0, vmax=1,
                        extent=[0, len(t) - 1, 0, sorted_act.shape[1]])
    axes[1].set_ylabel(ylabel, fontsize=8)
    axes[1].set_xticks([])
    fig.colorbar(im, ax=axes[1], fraction=0.02, pad=0.01, label='|tanh(h)|')

    # --- Bottom: GT spikes + predicted rate ---
    gt_times = np.where(y_binned > 0)[0]
    axes[2].vlines(gt_times, 0, 1, color='k', lw=1.0, alpha=0.7)
    ax2r = axes[2].twinx()
    ax2r.plot(t, y_pred, color=color, lw=1.2, alpha=0.7)
    ax2r.set_ylabel('Rate', fontsize=8)
    ax2r.set_ylim(bottom=0)
    axes[2].set_ylim(0, 1.5)
    axes[2].set_yticks([])
    axes[2].set_xlabel('Time (bins)')
    axes[2].set_xlim(0, len(t) - 1)
    axes[2].grid(alpha=0.2)

    fig.suptitle(f'{model_name} — Hidden Activations (Cell Type {cell_type})',
                 fontweight='bold')
    plt.tight_layout()
    _save(fig, get_plot_dir(cell_type, plot_root) / f'hidden_heatmap_{model_name}.png')


def plot_clean_comparison(trainers: Dict, I_binned: np.ndarray,
                          y_binned: np.ndarray, cell_type: int, plot_root: Path = None):
    """
    Single clean test trial: GT spikes (top) + one row per model showing
    Poisson-sampled spikes, all color-coded for quick visual comparison.
    """
    n_models = len(trainers)
    fig, axes = plt.subplots(n_models + 1, 1, figsize=(14, 1.5 * (n_models + 1)),
                             sharex=True)

    t = np.arange(len(I_binned))
    X_test = I_binned.reshape(1, -1, 1)
    gt_times = np.where(y_binned > 0)[0]

    # Top: stimulus shading + GT spikes
    ax0 = axes[0]
    ax0_tw = ax0.twinx()
    ax0_tw.fill_between(t, 0, I_binned / (np.abs(I_binned).max() + 1e-8),
                        color='cornflowerblue', alpha=0.15)
    ax0_tw.set_yticks([])
    ax0.vlines(gt_times, 0, 1, color='k', lw=1.5)
    ax0.set_ylim(0, 1.2)
    ax0.set_yticks([])
    ax0.set_ylabel('Ground Truth', fontsize=9, fontweight='bold')
    ax0.set_title(f'Clean Test Trial — Cell Type {cell_type}', fontweight='bold')

    for i, (name, trainer) in enumerate(trainers.items()):
        ax = axes[i + 1]
        color = MODEL_COLORS.get(name, 'gray')

        y_pred = trainer.predict(X_test)[0]
        y_sampled = np.random.poisson(y_pred)
        pred_times = np.where(y_sampled > 0)[0]

        ax.vlines(pred_times, 0, 1, color=color, lw=1.5, alpha=0.85)
        ax.set_ylim(0, 1.2)
        ax.set_yticks([])
        ax.set_ylabel(name, fontsize=8, fontweight='bold')

    axes[-1].set_xlabel('Time (bins)')
    plt.tight_layout()
    _save(fig, get_plot_dir(cell_type, plot_root) / 'clean_comparison.png')
