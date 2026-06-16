"""
run.py - Main entry point for training and evaluating RNN models on Izhikevich data.

Usage:
    python run.py                          # Default: cell types 1 2 3 4, parallel
    python run.py 3                        # Tonic bursting
    python run.py 1 3 4                    # Multiple cell types
    python run.py all                      # All available cell types
    python run.py --mode sequential 1 2    # Sequential baseline for benchmarking
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')

torch.set_num_threads(1)

import itertools
from pathlib import Path
from joblib import Parallel, delayed


def get_physical_cores() -> int:
    try:
        import psutil
        cores = psutil.cpu_count(logical=False)
        if cores is not None:
            return cores
    except ImportError:
        pass
    logical = os.cpu_count() or 2
    return max(1, logical // 2)
from models import build_model, MODEL_REGISTRY, MEMBRANE_ONLY_MODELS
from data import generate_training_data, generate_test_trial, generate_clean_test_trial
from train import Trainer, train_all_models
from plotting import (PLOT_ROOT, plot_training_data, plot_loss_curves, plot_test_predictions,
                      plot_tau_distributions, plot_clean_comparison, plot_hidden_activations,
                      plot_isi_distributions)
from evaluate import evaluate_all, plot_metric_heatmap  # noqa: E402
from izhikevich_configs import index_to_name


# =============================================================================
# Configuration
# =============================================================================

N_TRIALS = 1000
N_EPOCHS = 300
LR = 1e-3
BATCH_SIZE = 64
BIN_SIZE_MS = 1.0
N_TEST_TRIALS = 5
SEED = 42

MODEL_ROOT = Path(__file__).parent / 'models'

AVAILABLE_TYPES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21]


# =============================================================================
# Parallel worker
# =============================================================================

def _train_single_model_worker(ct, mname, X, y, lengths, hidden_size, tau_lr_multiplier, activation='tanh', view='membrane_potential', aux_variance_gamma=0.0):
    """Train a single (cell_type, model, grid_point) tuple — designed for joblib dispatch."""
    model = build_model(mname, input_size=1, hidden_size=hidden_size, dt=BIN_SIZE_MS, activation=activation, view=view)
    trainer = Trainer(model, lr=LR, batch_size=BATCH_SIZE, device='cpu',
                      tau_lr_multiplier=tau_lr_multiplier)
    trainer.fit(X, y, n_epochs=N_EPOCHS, lengths=lengths, verbose=False,
                aux_variance_gamma=aux_variance_gamma)
    return (ct, mname, hidden_size, tau_lr_multiplier, trainer)


# =============================================================================
# Plotting helper (shared by both modes)
# =============================================================================

def _plot_all(trainers, train_data, cell_type, plot_root: Path = None):
    """Generate all plots for a single cell type into plot_root/ct{N}_name/."""
    X, y, lengths = train_data

    plot_training_data(X, y, lengths, cell_type, plot_root=plot_root)
    plot_loss_curves(trainers, cell_type, plot_root=plot_root)

    test_trials = [generate_test_trial(cell_type, bin_size_ms=BIN_SIZE_MS)
                   for _ in range(N_TEST_TRIALS)]
    for k, (Ib, yb) in enumerate(test_trials):
        print(f"  Trial {k+1}: {len(Ib)} bins, {yb.sum():.0f} spikes")
    plot_test_predictions(trainers, test_trials, cell_type, plot_root=plot_root)
    plot_isi_distributions(trainers, test_trials, cell_type, plot_root=plot_root)
    plot_tau_distributions(trainers, cell_type, plot_root=plot_root)

    mean_pre_ms = 30.0
    mean_post_ms = 30.0
    mean_step_ms = max(100.0, lengths.mean() * BIN_SIZE_MS - mean_pre_ms - mean_post_ms)
    I_clean, y_clean = generate_clean_test_trial(
        cell_type, bin_size_ms=BIN_SIZE_MS,
        pre_ms=mean_pre_ms, step_ms=mean_step_ms, post_ms=mean_post_ms,
    )
    print(f"  Clean trial: {len(I_clean)} bins, {y_clean.sum():.0f} spikes "
          f"(pre={mean_pre_ms:.0f}ms, step={mean_step_ms:.0f}ms, post={mean_post_ms:.0f}ms)")
    plot_clean_comparison(trainers, I_clean, y_clean, cell_type, plot_root=plot_root)

    for mname, trainer in trainers.items():
        plot_hidden_activations(trainer, I_clean, y_clean, cell_type, mname, plot_root=plot_root)


# =============================================================================
# Main
# =============================================================================

def main():
    default_cores = get_physical_cores()
    parser = argparse.ArgumentParser(description='Train RNN models on Izhikevich data.')
    parser.add_argument('cell_types', nargs='*', default=['1', '2', '3', '4'],
                        help='Cell type indices or "all"')
    parser.add_argument('--mode', choices=['sequential', 'parallel'], default='parallel',
                        help='Training mode (default: parallel)')
    parser.add_argument('--n_jobs', type=int, default=default_cores,
                        help=f'Number of parallel workers (default: {default_cores} physical cores)')
    parser.add_argument('--hidden_size', type=int, nargs='+', default=[64],
                        help='Hidden unit counts to scan (e.g. --hidden_size 64 128 256)')
    parser.add_argument('--tau_lr_multiplier', type=float, nargs='+', default=[1.0],
                        help='tau LR multipliers to scan (e.g. --tau_lr_multiplier 1.0 5.0 10.0)')
    parser.add_argument('--save_models', action='store_true', default=True,
                        help='Save trained models to models/{run_tag}/ (default: True)')
    parser.add_argument('--no_save_models', dest='save_models', action='store_false',
                        help='Disable model saving')
    parser.add_argument('--skip_training', action='store_true', default=False,
                        help='Load saved models instead of training (requires prior --save_models run)')
    parser.add_argument('--activation', type=str, choices=['tanh', 'relu', 'softplus'], default='tanh',
                        help='Activation function for RNN models (default: tanh)')
    parser.add_argument('--view', type=str, choices=['membrane_potential', 'firing_rate'],
                        default='membrane_potential',
                        help='Hidden state update view (default: membrane_potential)')
    parser.add_argument('--aux_variance_gamma', type=float, default=0.0,
                        help='Auxiliary temporal variance penalty weight (default: 0.0 = disabled)')
    args = parser.parse_args()

    # Parse cell types
    if len(args.cell_types) == 1 and args.cell_types[0].lower() == 'all':
        valid_types = AVAILABLE_TYPES
    else:
        valid_types = [int(x) for x in args.cell_types]
    valid_types = [ct for ct in valid_types if ct in AVAILABLE_TYPES]

    if not valid_types:
        print("No valid cell types specified.")
        return

    # Filter models: firing_rate view excludes membrane-only models (Clockwork, PLRNN)
    active_models = {k: v for k, v in MODEL_REGISTRY.items()
                     if args.view == 'membrane_potential' or k not in MEMBRANE_ONLY_MODELS}

    grid = list(itertools.product(args.hidden_size, args.tau_lr_multiplier))
    n_models = len(active_models)
    n_tasks_per_point = len(valid_types) * n_models
    n_tasks_total = n_tasks_per_point * len(grid)

    print(f"Cell types:          {valid_types}")
    print(f"Models:              {list(active_models.keys())}")
    print(f"hidden_size grid:    {args.hidden_size}")
    print(f"tau_lr_mult grid:    {args.tau_lr_multiplier}")
    print(f"Activation:          {args.activation}")
    print(f"View:                {args.view}")
    print(f"Grid points:         {len(grid)}  ({args.hidden_size} x {args.tau_lr_multiplier})")
    print(f"Total tasks:         {n_tasks_total}  ({n_tasks_per_point} per grid point)")
    print(f"Mode:                {args.mode} (n_jobs={args.n_jobs})\n")

    act_root = PLOT_ROOT / args.view / args.activation
    model_act_root = MODEL_ROOT / args.view / args.activation

    def _model_path(run_tag, ct, mname) -> Path:
        ct_name = index_to_name.get(ct, f'type_{ct}').lower().replace(' ', '_')
        return model_act_root / run_tag / f'ct{ct}_{ct_name}' / f'{mname}.pt'

    # ---- Generate training data once (independent of grid) ----
    print("Generating training data...")
    train_data = {}
    test_data  = {}
    for ct in valid_types:
        ct_name = index_to_name.get(ct, f'Type {ct}')
        print(f"  Cell type {ct} ({ct_name})...")
        X, y, lengths = generate_training_data(
            ct, n_trials=N_TRIALS, bin_size_ms=BIN_SIZE_MS, seed=SEED, verbose=True,
        )
        train_data[ct] = (X[:-N_TEST_TRIALS], y[:-N_TEST_TRIALS], lengths[:-N_TEST_TRIALS])
        test_data[ct]  = (X[-N_TEST_TRIALS:], y[-N_TEST_TRIALS:], lengths[-N_TEST_TRIALS:])

    # ---- Load or train across full grid ----
    start_time = time.perf_counter()
    trainers_by_config = {}

    if args.skip_training:
        print("\nLoading saved models...")
        for hidden_size, tau_lr in grid:
            run_tag = f"h{hidden_size}_tlr{tau_lr}"
            trainers_by_config[(hidden_size, tau_lr)] = {}
            for ct in valid_types:
                trainers_by_config[(hidden_size, tau_lr)][ct] = {}
                for mname in active_models:
                    path = _model_path(run_tag, ct, mname)
                    if not path.exists():
                        raise FileNotFoundError(
                            f"No saved model at {path}. Run without --skip_training first."
                        )
                    model = build_model(mname, input_size=1, hidden_size=hidden_size, dt=BIN_SIZE_MS, activation=args.activation, view=args.view)
                    trainer = Trainer.load(path, model, device='cpu')
                    trainers_by_config[(hidden_size, tau_lr)][ct][mname] = trainer
                    print(f"  Loaded {run_tag}/ct{ct}/{mname}")

    elif args.mode == 'sequential':
        for hidden_size, tau_lr in grid:
            run_tag = f"h{hidden_size}_tlr{tau_lr}"
            print(f"\n{'='*60}")
            print(f"  Grid point: hidden_size={hidden_size}, tau_lr_multiplier={tau_lr}")
            print(f"{'='*60}")
            trainers_by_config[(hidden_size, tau_lr)] = {}
            for ct in valid_types:
                X, y, lengths = train_data[ct]
                ct_name = index_to_name.get(ct, f'Type {ct}')
                print(f"\n  --- Cell Type {ct}: {ct_name} ---")
                models = {mname: build_model(mname, input_size=1, hidden_size=hidden_size, dt=BIN_SIZE_MS, activation=args.activation, view=args.view)
                          for mname in active_models}
                for mname, m in models.items():
                    print(f"    {mname:20s}  params={sum(p.numel() for p in m.parameters()):,d}")
                trainers = train_all_models(
                    models, X, y, lengths=lengths, n_epochs=N_EPOCHS, lr=LR,
                    batch_size=BATCH_SIZE, verbose=True, tau_lr_multiplier=tau_lr,
                    aux_variance_gamma=args.aux_variance_gamma,
                )
                trainers_by_config[(hidden_size, tau_lr)][ct] = trainers
                if args.save_models:
                    for mname, trainer in trainers.items():
                        trainer.save(_model_path(run_tag, ct, mname))

    else:  # parallel — dispatch ALL grid × cell × model tasks at once
        all_tasks = [
            (ct, mname, hs, tlr)
            for hs, tlr in grid
            for ct in valid_types
            for mname in active_models
        ]
        print(f"\nDispatching {n_tasks_total} jobs with joblib (n_jobs={args.n_jobs})...")
        results = Parallel(n_jobs=args.n_jobs, verbose=10)(
            delayed(_train_single_model_worker)(
                ct, mname, train_data[ct][0], train_data[ct][1], train_data[ct][2], hs, tlr, args.activation, args.view, args.aux_variance_gamma
            )
            for ct, mname, hs, tlr in all_tasks
        )
        for ct, mname, hs, tlr, trainer in results:
            trainers_by_config.setdefault((hs, tlr), {}).setdefault(ct, {})[mname] = trainer
        if args.save_models:
            print("\nSaving models...")
            for (hs, tlr), by_ct in trainers_by_config.items():
                run_tag = f"h{hs}_tlr{tlr}"
                for ct, trainers in by_ct.items():
                    for mname, trainer in trainers.items():
                        trainer.save(_model_path(run_tag, ct, mname))

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    if not args.skip_training:
        print(f"\n{'#'*60}")
        print(f"  Training completed in {elapsed:.2f}s using {args.mode} mode")
        print(f"  {n_tasks_total} tasks, {elapsed/n_tasks_total:.2f}s avg per task")
        print(f"{'#'*60}")

    # ---- Plot (one directory per grid point) ----
    print("\nGenerating plots...")
    for (hidden_size, tau_lr), trainers_by_ct in trainers_by_config.items():
        run_tag = f"h{hidden_size}_tlr{tau_lr}"
        plot_root = act_root / run_tag
        print(f"\n  [{run_tag}]")
        for ct in valid_types:
            ct_name = index_to_name.get(ct, f'Type {ct}')
            print(f"    Cell type {ct} ({ct_name})...")
            _plot_all(trainers_by_ct[ct], train_data[ct], ct, plot_root=plot_root)

    # ---- Evaluate ----
    print("\nEvaluating models...")
    df = evaluate_all(trainers_by_config, test_data)

    # Save per-grid-point CSVs and heatmaps — each goes into its own run_tag directory
    for (hs, tlr) in trainers_by_config:
        run_tag = f"h{hs}_tlr{tlr}"
        plot_root = act_root / run_tag
        plot_root.mkdir(parents=True, exist_ok=True)
        sub = df[(df['hidden_size'] == hs) & (df['tau_lr'] == tlr)]
        csv_path = plot_root / 'metrics.csv'
        sub.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")
        for metric in ('bps', 'pearson_r', 'test_nll'):
            plot_metric_heatmap(sub, metric=metric, plot_root=plot_root)

    print(f"\nDone! Results saved to {act_root}/")


if __name__ == '__main__':
    main()
