"""
benchmark.py - Measure sequential vs parallel training speed.

Runs both modes on the same cell types (no plotting) and writes results
to benchmark_results.txt.

Usage:
    python benchmark.py              # Default: cell types 1 2 3 4
    python benchmark.py 1 2          # Specific types
    python benchmark.py all          # All 19 types
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import time
import platform
import subprocess
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')

torch.set_num_threads(1)

from joblib import Parallel, delayed, cpu_count
from models import build_model, MODEL_REGISTRY
from data import generate_training_data
from train import Trainer, train_all_models
from izhikevich_configs import index_to_name

# Mirror run.py config
HIDDEN_SIZE = 64
N_TRIALS = 1000
N_EPOCHS = 300
LR = 1e-3
BATCH_SIZE = 64
BIN_SIZE_MS = 1.0
SEED = 42

AVAILABLE_TYPES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21]


def _worker(ct, mname, X, y, lengths):
    model = build_model(mname, input_size=1, hidden_size=HIDDEN_SIZE, dt=BIN_SIZE_MS)
    trainer = Trainer(model, lr=LR, batch_size=BATCH_SIZE, device='cpu')
    t0 = time.perf_counter()
    trainer.fit(X, y, n_epochs=N_EPOCHS, lengths=lengths, verbose=False)
    return (ct, mname, time.perf_counter() - t0)


def _get_cpu_name():
    try:
        out = subprocess.check_output("lscpu", text=True)
        for line in out.splitlines():
            if "Model name" in line:
                return line.split(":")[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark sequential vs parallel training.')
    parser.add_argument('cell_types', nargs='*', default=['1', '2', '3', '4'],
                        help='Cell type indices or "all"')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Number of parallel workers (default: -1 for all logical cores)')
    args = parser.parse_args()

    if len(args.cell_types) == 1 and args.cell_types[0].lower() == 'all':
        cell_types = AVAILABLE_TYPES
    else:
        cell_types = [int(x) for x in args.cell_types]
    cell_types = [ct for ct in cell_types if ct in AVAILABLE_TYPES]

    n_models = len(MODEL_REGISTRY)
    n_tasks = len(cell_types) * n_models
    n_cores = cpu_count()

    print(f"Benchmark: {len(cell_types)} cell types x {n_models} models = {n_tasks} tasks")
    print(f"CPU: {_get_cpu_name()} ({n_cores} logical cores)")
    print(f"Config: {N_TRIALS} trials, {N_EPOCHS} epochs, hidden={HIDDEN_SIZE}\n")

    # Generate data once
    print("Generating data...")
    train_data = {}
    for ct in cell_types:
        X, y, lengths = generate_training_data(
            ct, n_trials=N_TRIALS, bin_size_ms=BIN_SIZE_MS, seed=SEED, verbose=False,
        )
        train_data[ct] = (X, y, lengths)
    print("Data ready.\n")

    # --- Sequential ---
    print("=" * 50)
    print("Running SEQUENTIAL...")
    print("=" * 50)
    seq_times = {}
    t_seq_start = time.perf_counter()
    for ct in cell_types:
        X, y, lengths = train_data[ct]
        models = {mname: build_model(mname, input_size=1, hidden_size=HIDDEN_SIZE, dt=BIN_SIZE_MS)
                  for mname in MODEL_REGISTRY}
        for mname, model in models.items():
            trainer = Trainer(model, lr=LR, batch_size=BATCH_SIZE, device='cpu')
            t0 = time.perf_counter()
            trainer.fit(X, y, n_epochs=N_EPOCHS, lengths=lengths, verbose=False)
            elapsed = time.perf_counter() - t0
            seq_times[(ct, mname)] = elapsed
            print(f"  ct={ct:2d} {mname:22s} {elapsed:6.1f}s")
    t_seq = time.perf_counter() - t_seq_start
    print(f"Sequential total: {t_seq:.1f}s\n")

    # --- Parallel ---
    print("=" * 50)
    print(f"Running PARALLEL (n_jobs={args.n_jobs})...")
    print("=" * 50)
    tasks = [(ct, mname) for ct in cell_types for mname in MODEL_REGISTRY]
    t_par_start = time.perf_counter()
    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(_worker)(ct, mname, train_data[ct][0], train_data[ct][1], train_data[ct][2])
        for ct, mname in tasks
    )
    t_par = time.perf_counter() - t_par_start

    par_times = {}
    for ct, mname, elapsed in results:
        par_times[(ct, mname)] = elapsed
    print(f"Parallel total: {t_par:.1f}s\n")

    # --- Summary ---
    speedup = t_seq / t_par if t_par > 0 else float('inf')
    efficiency = speedup / n_cores * 100

    # Per-model average times
    model_seq_avg = {}
    model_par_avg = {}
    for mname in MODEL_REGISTRY:
        model_seq_avg[mname] = np.mean([seq_times[(ct, mname)] for ct in cell_types])
        model_par_avg[mname] = np.mean([par_times[(ct, mname)] for ct in cell_types])

    # Build report
    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"BENCHMARK RESULTS")
    lines.append(f"{'='*60}")
    lines.append(f"Date:       {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"CPU:        {_get_cpu_name()}")
    lines.append(f"Cores:      {n_cores} logical")
    lines.append(f"PyTorch:    {torch.__version__}")
    lines.append(f"Python:     {platform.python_version()}")
    lines.append(f"")
    lines.append(f"Config:")
    lines.append(f"  Cell types: {cell_types}")
    lines.append(f"  Models:     {n_models}")
    lines.append(f"  Tasks:      {n_tasks}")
    lines.append(f"  n_jobs:     {args.n_jobs}")
    lines.append(f"  Trials:     {N_TRIALS}")
    lines.append(f"  Epochs:     {N_EPOCHS}")
    lines.append(f"  Hidden:     {HIDDEN_SIZE}")
    lines.append(f"")
    lines.append(f"{'='*60}")
    lines.append(f"TIMING")
    lines.append(f"{'='*60}")
    lines.append(f"  Sequential:  {t_seq:8.1f}s")
    lines.append(f"  Parallel:    {t_par:8.1f}s")
    lines.append(f"  Speedup:     {speedup:8.2f}x")
    lines.append(f"  Efficiency:  {efficiency:7.1f}% (speedup / {n_cores} cores)")
    lines.append(f"")
    lines.append(f"{'='*60}")
    lines.append(f"PER-MODEL AVERAGE (sequential)")
    lines.append(f"{'='*60}")
    for mname in MODEL_REGISTRY:
        lines.append(f"  {mname:22s}  {model_seq_avg[mname]:6.1f}s")
    lines.append(f"")
    lines.append(f"{'='*60}")
    lines.append(f"PER-TASK DETAIL (sequential)")
    lines.append(f"{'='*60}")
    for ct in cell_types:
        for mname in MODEL_REGISTRY:
            lines.append(f"  ct={ct:2d}  {mname:22s}  {seq_times[(ct, mname)]:6.1f}s")

    report = "\n".join(lines)
    print(f"\n{report}")

    out_path = os.path.join(os.path.dirname(__file__), "benchmark_results.txt")
    with open(out_path, "w") as f:
        f.write(report + "\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
