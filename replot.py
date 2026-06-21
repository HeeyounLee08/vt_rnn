"""
replot.py - Regenerate all plots for every saved model without re-training.

Auto-discovers all saved models under models/{view}/{activation}/h{N}_tlr{M}/ct{X}_{name}/{Model}.pt
and re-runs the full plotting + evaluation pipeline.

Usage:
    python replot.py                                    # Everything
    python replot.py --view membrane_potential          # One view, all activations/sizes
    python replot.py --activation relu                  # One activation, all views/sizes
    python replot.py --hidden_size 64 128               # Specific hidden sizes only
    python replot.py --view membrane_potential --activation relu --hidden_size 64
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import re
import argparse
import matplotlib
matplotlib.use('Agg')

from collections import defaultdict
from pathlib import Path

from models import build_model
from data import generate_training_data, generate_test_trial, generate_clean_test_trial
from train import Trainer
from plotting import (PLOT_ROOT, plot_training_data, plot_loss_curves, plot_test_predictions,
                      plot_tau_distributions, plot_clean_comparison, plot_hidden_activations,
                      plot_isi_distributions, plot_rate_autocorrelation)
from evaluate import evaluate_all, plot_metric_heatmap
from izhikevich_configs import index_to_name

# Keep in sync with run.py
N_TRIALS      = 1000
N_TEST_TRIALS = 5
BIN_SIZE_MS   = 1.0
SEED          = 42

MODEL_ROOT   = Path(__file__).parent / 'models'
RESULTS_ROOT = Path(__file__).parent / 'results'


# ── Path parsing ──────────────────────────────────────────────────────────────

def _parse_run_tag(tag: str):
    """'h64_tlr5.0' → (64, 5.0, SEED)   'h128_tlr1.0_seed42' → (128, 1.0, 42)"""
    m = re.fullmatch(r'h(\d+)_tlr([\d.]+)(?:_seed(\d+))?', tag)
    if not m:
        raise ValueError(f"Unexpected run_tag: {tag!r}")
    seed = int(m.group(3)) if m.group(3) else SEED
    return int(m.group(1)), float(m.group(2)), seed


def _parse_ct_dir(name: str):
    """'ct1_tonic_spiking' → 1"""
    m = re.match(r'ct(\d+)', name)
    if not m:
        raise ValueError(f"Unexpected ct_dir: {name!r}")
    return int(m.group(1))


def scan_models(view=None, activation=None, hidden_sizes=None):
    """
    Scan models/ and return a list of entries, each a dict:
      {view, activation, hidden_size, tau_lr, seed, run_tag, cell_type, model_name, path}
    All arguments act as optional filters.
    """
    entries = []
    for pt in sorted(MODEL_ROOT.rglob('*.pt')):
        try:
            parts = pt.relative_to(MODEL_ROOT).parts
            # Expected depth: view / activation / run_tag / ct_dir / Model.pt
            if len(parts) != 5:
                continue
            v, act, run_tag, ct_dir, fname = parts
            hs, tlr, seed = _parse_run_tag(run_tag)
            ct = _parse_ct_dir(ct_dir)
        except Exception:
            continue

        if view       and v   != view:       continue
        if activation and act != activation: continue
        if hidden_sizes and hs not in hidden_sizes: continue

        entries.append(dict(
            view=v, activation=act,
            hidden_size=hs, tau_lr=tlr, seed=seed, run_tag=run_tag,
            cell_type=ct, model_name=Path(fname).stem,
            path=pt,
        ))
    return entries


# ── Plotting helper ───────────────────────────────────────────────────────────

def _run_plots(trainers, X, y, lengths, cell_type, plot_root):
    plot_training_data(X, y, lengths, cell_type, plot_root=plot_root)
    plot_loss_curves(trainers, cell_type, plot_root=plot_root)

    test_trials = [generate_test_trial(cell_type, bin_size_ms=BIN_SIZE_MS)
                   for _ in range(N_TEST_TRIALS)]
    for k, (Ib, yb) in enumerate(test_trials):
        print(f"      trial {k+1}: {len(Ib)} bins, {int(yb.sum())} spikes")
    plot_test_predictions(trainers, test_trials, cell_type, plot_root=plot_root)
    plot_isi_distributions(trainers, test_trials, cell_type, plot_root=plot_root)
    plot_rate_autocorrelation(trainers, test_trials, cell_type, plot_root=plot_root)
    plot_tau_distributions(trainers, cell_type, plot_root=plot_root)

    pre_ms, post_ms = 30.0, 30.0
    step_ms = max(100.0, lengths.mean() * BIN_SIZE_MS - pre_ms - post_ms)
    I_clean, y_clean = generate_clean_test_trial(
        cell_type, bin_size_ms=BIN_SIZE_MS,
        pre_ms=pre_ms, step_ms=step_ms, post_ms=post_ms,
    )
    print(f"      clean trial: {len(I_clean)} bins, {int(y_clean.sum())} spikes")
    plot_clean_comparison(trainers, I_clean, y_clean, cell_type, plot_root=plot_root)
    for mname, trainer in trainers.items():
        plot_hidden_activations(trainer, I_clean, y_clean, cell_type, mname,
                                plot_root=plot_root)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Replot all saved models without re-training.')
    parser.add_argument('--view',        type=str,          default=None)
    parser.add_argument('--activation',  type=str,          default=None)
    parser.add_argument('--hidden_size', type=int, nargs='+', default=None)
    args = parser.parse_args()

    entries = scan_models(args.view, args.activation, args.hidden_size)
    if not entries:
        print("No saved models found matching the specified filters.")
        return

    va_pairs   = sorted(set((e['view'], e['activation']) for e in entries))
    cell_types = sorted(set(e['cell_type'] for e in entries))
    grid_pts   = sorted(set((e['hidden_size'], e['tau_lr'], e['seed']) for e in entries))
    print(f"Found {len(entries)} saved model files")
    print(f"  View/activation pairs    : {va_pairs}")
    print(f"  Cell types               : {cell_types}")
    print(f"  Grid points (h, τ_lr, s) : {grid_pts}")

    # Regenerate training data with the original seed
    print("\nRegenerating training data...")
    train_data = {}
    test_data  = {}
    for ct in cell_types:
        ct_name = index_to_name.get(ct, f'Type {ct}')
        print(f"  Cell type {ct} ({ct_name})...")
        X, y, lengths = generate_training_data(
            ct, n_trials=N_TRIALS, bin_size_ms=BIN_SIZE_MS, seed=SEED, verbose=True,
        )
        train_data[ct] = (X[:-N_TEST_TRIALS], y[:-N_TEST_TRIALS], lengths[:-N_TEST_TRIALS])
        test_data[ct]  = (X[-N_TEST_TRIALS:], y[-N_TEST_TRIALS:], lengths[-N_TEST_TRIALS:])

    # Group entries: by_va[(view, act)][(hs, tlr, seed)][ct][mname] = (path, run_tag)
    by_va = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    run_tag_for = {}  # (view, act, hs, tlr, seed) → original run_tag directory name
    for e in entries:
        key = (e['hidden_size'], e['tau_lr'], e['seed'])
        by_va[(e['view'], e['activation'])][key][e['cell_type']][e['model_name']] = e['path']
        run_tag_for[(e['view'], e['activation'], *key)] = e['run_tag']

    # Process each (view, activation) group independently
    for (view, act), by_grid in sorted(by_va.items()):
        act_root = RESULTS_ROOT / view / act
        print(f"\n{'='*64}")
        print(f"  view={view}  activation={act}")
        print(f"{'='*64}")

        trainers_by_config = defaultdict(dict)  # for evaluate_all

        for (hs, tlr, seed), by_ct in sorted(by_grid.items()):
            run_tag   = run_tag_for[(view, act, hs, tlr, seed)]
            plot_root = act_root / run_tag
            print(f"\n  [{run_tag}]")

            for ct, model_paths in sorted(by_ct.items()):
                ct_name  = index_to_name.get(ct, f'Type {ct}')
                print(f"    Cell type {ct} ({ct_name}) — loading {list(model_paths.keys())}...")

                trainers = {}
                for mname, path in sorted(model_paths.items()):
                    try:
                        model   = build_model(mname, input_size=1, hidden_size=hs,
                                              dt=BIN_SIZE_MS, activation=act, view=view)
                        trainer = Trainer.load(path, model, device='cpu')
                        trainers[mname] = trainer
                    except Exception as ex:
                        print(f"      SKIP {mname}: {ex}")

                if not trainers:
                    continue

                trainers_by_config[(hs, tlr, seed)][ct] = trainers
                X, y, lengths = train_data[ct]
                _run_plots(trainers, X, y, lengths, ct, plot_root)

        # Evaluate and save metrics
        if trainers_by_config:
            print(f"\n  Evaluating...")
            df = evaluate_all(trainers_by_config, test_data)
            for (hs, tlr, seed) in trainers_by_config:
                run_tag   = run_tag_for[(view, act, hs, tlr, seed)]
                plot_root = act_root / run_tag
                plot_root.mkdir(parents=True, exist_ok=True)
                sub = df[(df['hidden_size'] == hs) & (df['tau_lr'] == tlr) & (df['seed'] == seed)]
                csv_path = plot_root / 'metrics.csv'
                sub.to_csv(csv_path, index=False)
                print(f"  Saved: {csv_path}")
                for metric in ('bps', 'pearson_r', 'test_nll'):
                    plot_metric_heatmap(sub, metric=metric, plot_root=plot_root)

    print("\nDone!")


if __name__ == '__main__':
    main()
