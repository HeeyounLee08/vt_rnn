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
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter1d
from typing import Dict
from pathlib import Path

import torch
from models import MODEL_COLORS
from izhikevich_configs import index_to_name

PLOT_ROOT = Path(__file__).parent / 'results'
PLOT_ROOT.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'legend.framealpha': 0.7,
    'figure.dpi': 150,
})


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

        if name in ('VanillaRNN', 'PLRNN'):
            label = 'No τ\n(discrete RNN)' if name == 'VanillaRNN' else 'No τ\n(diagonal A)'
            ax.text(0.5, 0.5, label, ha='center', va='center',
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
            taus = taus[np.isfinite(taus)]
            if len(taus) == 0:
                ax.text(0.5, 0.5, 'No finite τ', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12, color='gray')
                ax.set_axis_off()
            else:
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
            taus = taus[np.isfinite(taus)]
            if len(taus) == 0:
                ax.text(0.5, 0.5, 'No finite τ', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12, color='gray')
                ax.set_axis_off()
            else:
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

        elif name == 'PureLogTau':
            with torch.no_grad():
                alpha = model.get_alpha().cpu().numpy()
            taus = model.dt / alpha
            taus = taus[np.isfinite(taus)]
            if len(taus) == 0:
                ax.text(0.5, 0.5, 'No finite τ', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12, color='gray')
                ax.set_axis_off()
            else:
                ax.hist(taus, bins=20, color=color, alpha=0.8, edgecolor='k', lw=0.5)
                ax.axvline(taus.mean(), color='k', ls='--', lw=1.2,
                           label=f'mean={taus.mean():.1f}ms')
                ax.set_xlabel('τ (ms)')
                ax.set_ylabel('# units')
                ax.legend(fontsize=8)
            ax.set_title(f'{name}\n[log-space, per-unit]', fontsize=9)

        elif name == 'PureLogPartitioned':
            with torch.no_grad():
                alpha = model.get_alpha().cpu().numpy()
            taus = model.dt / alpha
            tau_fast = taus[:model.n_fast]
            tau_slow = taus[model.n_fast:]
            ax.bar([f'Fast\n(τ={tau_fast[0]:.1f}ms)',
                    f'Slow\n(τ={tau_slow[0]:.1f}ms)'],
                   [model.n_fast, model.n_slow],
                   color=[color, '#b2df8a'], edgecolor='k', lw=0.8)
            ax.set_ylabel('# units')
            ax.set_title(f'{name}\n[log-space, 2 params]', fontsize=9)

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
        y_pred = np.clip(np.nan_to_num(y_pred, nan=0.0, posinf=1e6), 0, 1e6)
        h_seq = h_seq.squeeze(0).cpu().numpy()  # (seq_len, hidden_size)

    act_name = getattr(model, 'activation_name', 'tanh')
    view = getattr(model, 'view', 'membrane_potential')
    if view == 'firing_rate':
        act = h_seq  # already post-activation
    elif act_name == 'relu':
        act = np.maximum(0, h_seq)
    else:
        act = np.tanh(h_seq)  # values in [-1, 1]

    # Sorting logic — retain metric for side panel
    if model_name in ('VanillaRNN', 'PLRNN'):
        variances = act.var(axis=0)
        sort_idx = np.argsort(variances)
        metric = variances[sort_idx]
        xlabel = 'Variance'
        ylabel = 'Unit (sorted by variance)'
    elif model_name == 'Clockwork':
        periods = model.unit_periods.cpu().numpy()
        sort_idx = np.argsort(periods)
        metric = periods[sort_idx].astype(float)
        xlabel = 'Period'
        ylabel = 'Unit (sorted by period)'
    else:
        if hasattr(model, 'get_alpha'):
            with torch.no_grad():
                alpha = model.get_alpha().cpu().numpy()
        else:
            alpha = model.alpha.cpu().numpy()
        taus = model.dt / alpha
        sort_idx = np.argsort(taus)
        metric = taus[sort_idx]
        xlabel = 'Tau (ms)'
        ylabel = 'Unit (sorted by τ)'

    sorted_act = act[:, sort_idx]  # (seq_len, hidden_size)
    n_units = sorted_act.shape[1]
    t = np.arange(len(I_binned))
    color = MODEL_COLORS.get(model_name, 'gray')

    # 3-column GridSpec:
    #   col 0: timescale side panel (heatmap row only)
    #   col 1: all three time-aligned panels (sharex keeps them locked)
    #   col 2: colorbar (heatmap row only) — dedicated column avoids squishing col 1
    fig = plt.figure(figsize=(14, 6), constrained_layout=True)
    gs = GridSpec(3, 3, figure=fig,
                  height_ratios=[1, 3, 1],
                  width_ratios=[1, 6, 0.12])

    ax_side    = fig.add_subplot(gs[1, 0])
    ax_stim    = fig.add_subplot(gs[0, 1])
    ax_heatmap = fig.add_subplot(gs[1, 1], sharex=ax_stim)
    ax_out     = fig.add_subplot(gs[2, 1], sharex=ax_stim)
    ax_cbar    = fig.add_subplot(gs[1, 2])

    # --- Side panel (timescales) ---
    y_coords = np.arange(n_units)
    ax_side.set_ylim(-0.5, n_units - 0.5)
    ax_side.set_yticks([])
    ax_side.set_ylabel(ylabel, fontsize=8)
    if model_name in ('VanillaRNN', 'PLRNN'):
        ax_side.set_axis_off()
    else:
        ax_side.fill_betweenx(y_coords, 0, metric, color='gray', alpha=0.6)
        ax_side.plot(metric, y_coords, color='k', lw=1)
        ax_side.set_xlabel(xlabel, fontsize=8)
        ax_side.tick_params(axis='x', labelsize=7)
        ax_side.set_xscale('log')
        ax_side.spines['left'].set_visible(False)
        ax_side.grid(axis='x', alpha=0.2)

    # --- Top: stimulus (no x-axis) ---
    ax_stim.plot(t, I_binned, color='cornflowerblue', lw=1.2)
    ax_stim.set_ylabel('Input', fontsize=8)
    ax_stim.set_xlim(0, len(t) - 1)
    ax_stim.spines['bottom'].set_visible(False)
    ax_stim.tick_params(bottom=False, labelbottom=False)
    ax_stim.grid(axis='y', alpha=0.2)

    # --- Middle: heatmap (no x-axis) ---
    if act_name == 'relu':
        vmax_act = np.percentile(sorted_act, 99) or 1.0
        im = ax_heatmap.imshow(sorted_act.T, aspect='auto', origin='lower',
                               cmap='magma', vmin=0, vmax=vmax_act,
                               extent=[0, len(t) - 1, -0.5, n_units - 0.5])
        cbar_label = 'ReLU(h)'
    else:
        im = ax_heatmap.imshow(sorted_act.T, aspect='auto', origin='lower',
                               cmap='RdBu_r', vmin=-1, vmax=1,
                               extent=[0, len(t) - 1, -0.5, n_units - 0.5])
        cbar_label = 'tanh(h)'
    ax_heatmap.set_ylim(-0.5, n_units - 0.5)
    ax_heatmap.set_yticks([])
    ax_heatmap.spines['bottom'].set_visible(False)
    ax_heatmap.tick_params(bottom=False, labelbottom=False)
    fig.colorbar(im, cax=ax_cbar, label=cbar_label)

    # --- Bottom: GT spikes + predicted spikes + rate ---
    gt_times = np.where(y_binned > 0)[0]
    pred_times = np.where(np.random.poisson(y_pred) > 0)[0]
    ax_out.vlines(gt_times,   1.1, 1.9, color='k',    lw=1.2)
    ax_out.vlines(pred_times, 0.1, 0.9, color=color,  lw=1.2, alpha=0.8)
    ax_out.set_yticks([0.5, 1.5])
    ax_out.set_yticklabels(['Pred', 'GT'], fontsize=8)
    ax_out_r = ax_out.twinx()
    gt_smooth = gaussian_filter1d(y_binned.astype(float), sigma=1.5)
    ax_out_r.plot(t, gt_smooth, color='k', lw=1.2, alpha=0.55, linestyle='-',
                  label='GT (smoothed)')
    ax_out_r.plot(t, y_pred, color=color, lw=1.2, alpha=0.5, linestyle='--',
                  label='Predicted')
    ax_out_r.set_ylabel('Rate', fontsize=8)
    ax_out_r.set_ylim(bottom=0)
    ax_out_r.spines['top'].set_visible(False)
    ax_out_r.legend(fontsize=7, loc='upper right')
    ax_out.set_ylim(0, 2)
    ax_out.set_xlabel('Time (bins)')
    ax_out.grid(axis='x', alpha=0.2)

    fig.suptitle(f'{model_name} — Hidden Activations (Cell Type {cell_type})',
                 fontweight='bold')
    _save(fig, get_plot_dir(cell_type, plot_root) / f'hidden_heatmap_{model_name}.png')


def _collect_isis(y_spikes: np.ndarray) -> np.ndarray:
    """ISIs (in bins) from a spike train; handles multi-spike bins by repeating."""
    spike_times = np.repeat(np.arange(len(y_spikes)), y_spikes.astype(int))
    if len(spike_times) < 2:
        return np.array([])
    return np.diff(spike_times).astype(float)


_ISI_HATCHES = ['///', '\\\\\\', '|||', '---', 'xxx', '+++', 'ooo', '...']


def plot_isi_distributions(trainers: Dict, test_trials: list,
                           cell_type: int, plot_root: Path = None):
    """
    ISI distribution: ground truth vs all models, across all test trials.
    Linear x-axis, fine bins, transparent hatched bars, color-coded per model.
    """
    gt_isis = np.concatenate([_collect_isis(yb) for _, yb in test_trials])
    gt_isis = gt_isis[gt_isis > 0]

    model_isis = {}
    for name, trainer in trainers.items():
        segs = []
        for I_binned, _ in test_trials:
            y_pred = trainer.predict(I_binned.reshape(1, -1, 1))[0]
            y_sampled = np.random.poisson(y_pred)
            seg = _collect_isis(y_sampled)
            if len(seg):
                segs.append(seg)
        model_isis[name] = np.concatenate(segs) if segs else np.array([])

    all_vals = np.concatenate([gt_isis] + [v for v in model_isis.values() if len(v)])
    all_vals = all_vals[all_vals > 0]
    if len(all_vals) < 2:
        return

    from scipy.ndimage import gaussian_filter1d

    x_max = np.percentile(all_vals, 98)
    bins = np.linspace(0, x_max, 80)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    smooth_sigma = 2.0

    def _smooth_density(isis):
        counts, _ = np.histogram(isis, bins=bins, density=True)
        return gaussian_filter1d(counts.astype(float), sigma=smooth_sigma)

    fig, ax = plt.subplots(figsize=(10, 5))

    if len(gt_isis) >= 2:
        sm = _smooth_density(gt_isis)
        ax.fill_between(bin_centers, sm, alpha=0.25, color='#333333',
                        hatch='', zorder=len(trainers) + 2)
        ax.plot(bin_centers, sm, color='#333333', linewidth=1.8,
                label=f'Ground Truth (n={len(gt_isis)})', zorder=len(trainers) + 2)

    for i, (name, isis) in enumerate(model_isis.items()):
        color = MODEL_COLORS.get(name, 'gray')
        hatch = _ISI_HATCHES[i % len(_ISI_HATCHES)]
        if len(isis) >= 2:
            sm = _smooth_density(isis)
            ax.fill_between(bin_centers, sm, alpha=0.18, color=color,
                            hatch=hatch, zorder=i + 1)
            ax.plot(bin_centers, sm, color=color, linewidth=1.4,
                    label=f'{name} (n={len(isis)})', zorder=i + 1)

    ax.set_xlim(0, x_max)
    ax.set_xlabel('ISI (bins)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(f'ISI Distribution — Cell Type {cell_type}', fontsize=11, fontweight='bold')
    ax.legend(ncol=2, fontsize=7, framealpha=0.85, loc='upper right')
    ax.grid(alpha=0.25, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    _save(fig, get_plot_dir(cell_type, plot_root) / 'isi_distributions.png')


def plot_rate_autocorrelation(trainers: Dict, test_trials: list,
                              cell_type: int, plot_root: Path = None,
                              max_lag: int = 100, expected_period: int = 27):
    """
    Autocorrelation of the predicted firing rate for each model, averaged across
    test trials. Compared to the GT autocorrelation (from smoothed spike trains).

    A peak at lag `expected_period` means the model's rate oscillates at that
    period — i.e., the model locked onto the limit-cycle rhythm, not just the
    mean rate.
    """
    def _acf(signal: np.ndarray, max_lag: int) -> np.ndarray:
        """Normalized autocorrelation at lags 1..max_lag (lag-0 excluded)."""
        x = signal - signal.mean()
        var = (x ** 2).mean()
        if var < 1e-12:
            return np.zeros(max_lag)
        T = len(x)
        result = np.array([
            (x[:T - k] * x[k:]).mean() / var
            for k in range(1, max_lag + 1)
        ])
        return result

    lags = np.arange(1, max_lag + 1)

    # GT autocorrelation: use Gaussian-smoothed spike train so sparsity doesn't
    # kill the signal — same sigma used throughout (3 bins ≈ 3 ms)
    gt_acfs = []
    for _, y_binned in test_trials:
        gt_smooth = gaussian_filter1d(y_binned.astype(float), sigma=3.0)
        gt_acfs.append(_acf(gt_smooth, max_lag))
    gt_acf_mean = np.mean(gt_acfs, axis=0)

    model_acfs = {}
    for name, trainer in trainers.items():
        trial_acfs = []
        for I_binned, _ in test_trials:
            y_pred = trainer.predict(I_binned.reshape(1, -1, 1))[0]
            trial_acfs.append(_acf(y_pred, max_lag))
        model_acfs[name] = np.mean(trial_acfs, axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(lags, gt_acf_mean, color='k', lw=2.2, label='Ground Truth (smoothed)',
            zorder=len(trainers) + 2)
    ax.fill_between(lags, 0, gt_acf_mean, color='k', alpha=0.08,
                    zorder=len(trainers) + 1)

    for i, (name, acf) in enumerate(model_acfs.items()):
        color = MODEL_COLORS.get(name, 'gray')
        ax.plot(lags, acf, color=color, lw=1.6, label=name, zorder=i + 1)

    # Mark expected period
    ax.axvline(expected_period, color='crimson', lw=1.2, linestyle='--', alpha=0.8,
               label=f'Expected period ({expected_period} bins)')
    ax.axhline(0, color='k', lw=0.6, alpha=0.4)

    ax.set_xlim(1, max_lag)
    ax.set_xlabel('Lag (bins)', fontsize=10)
    ax.set_ylabel('Autocorrelation', fontsize=10)
    ax.set_title(
        f'Rate Autocorrelation — Cell Type {cell_type}\n'
        f'Peak at lag {expected_period} = model learned the limit-cycle rhythm',
        fontsize=10, fontweight='bold',
    )
    ax.legend(ncol=2, fontsize=8, framealpha=0.85)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    _save(fig, get_plot_dir(cell_type, plot_root) / 'rate_autocorrelation.png')


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
