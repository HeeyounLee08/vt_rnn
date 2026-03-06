"""
data.py - Generate trial-based training data from Izhikevich neurons.

Produces binned spike count targets and corresponding current inputs
suitable for training sequence-to-sequence RNN models.

Adapted from rnn_data_generator.py in the GLM project.
"""

import numpy as np
from typing import Tuple, Optional

from generate_izhikevich_stim import generate_izhikevich_stim
from simulate_izhikevich import simulate_izhikevich


# Default parameters per cell type (amplitude, dt)
_CELL_PARAMS = {
    1: (14, 0.1), 2: (0.5, 0.1), 3: (10, 0.1), 4: (0.6, 0.1),
    5: (10, 0.1), 6: (20, 0.1), 7: (25, 0.1), 8: (0.5, 0.1),
    9: (3.49, 0.1), 11: (0.3, 0.5), 12: (27.4, 0.5),
    13: (-5, 0.1), 14: (-5, 0.1), 15: (2.3, 1.0), 16: (26.1, 0.05),
    18: (20, 0.1), 19: (70, 0.1), 20: (70, 0.1), 21: (26.1, 0.05),
}

# Baseline currents for cell types that don't rest at I=0
_BASELINE = {16: -65.0, 18: -70.0, 19: 80.0, 20: 80.0, 21: -65.0}

# For these types, the step amplitude IS the absolute current (not a delta)
_STEP_ABSOLUTE = {19, 20}

BURN_IN_MS = 500.0


def _ou_noise(n: int, dt: float, tau: float = 10.0, sigma: float = 1.0) -> np.ndarray:
    """Ornstein-Uhlenbeck colored noise."""
    noise = np.zeros(n)
    decay = np.exp(-dt / tau)
    scale = sigma * np.sqrt(1 - decay ** 2)
    for i in range(1, n):
        noise[i] = decay * noise[i - 1] + scale * np.random.randn()
    return noise


def _simulate_trial(cell_type: int, trial_duration_ms: float = 170.0,
                    silence_pre_ms: float = 20.0, step_duration_ms: float = 100.0,
                    jitter: bool = True, diverse_timing: bool = True,
                    ou_noise: bool = False,
                    ou_tau: float = 10.0, ou_sigma_fraction: float = 0.02,
                    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Simulate a single step-current trial through an Izhikevich neuron.

    A 500ms burn-in with baseline current is prepended so the neuron reaches
    equilibrium before the trial starts (critical for phasic types).

    Returns:
        I: current trace at simulation resolution (burn-in stripped)
        spike_times: spike times in ms (relative to trial start)
        dt: simulation timestep
        trial_duration_ms: actual trial duration (varies with diverse_timing)
    """
    amp, dt = _CELL_PARAMS.get(cell_type, (10.0, 0.1))
    baseline = _BASELINE.get(cell_type, 0.0)

    # Timing: diverse draws or fixed with small jitter
    if diverse_timing:
        pre = 20.0 + np.random.exponential(10.0)
        dur = float(np.clip(np.random.lognormal(np.log(100), 0.7), 100, 500))
        post = 20.0 + np.random.exponential(10.0)
        trial_duration_ms = pre + dur + post
    elif jitter:
        pre = silence_pre_ms + np.random.uniform(-5, 5)
        dur = step_duration_ms * np.random.uniform(0.95, 1.05)
        amp = amp * np.random.uniform(0.95, 1.05)
    else:
        pre, dur = silence_pre_ms, step_duration_ms

    burn_in_samples = int(BURN_IN_MS / dt)
    n_samples = int(trial_duration_ms / dt)

    # Build stimulus: burn-in (baseline) + trial (baseline with step)
    I = np.full(burn_in_samples + n_samples, baseline)
    start = burn_in_samples + int(max(0, pre) / dt)
    end = min(burn_in_samples + int((pre + dur) / dt), len(I))
    if cell_type in _STEP_ABSOLUTE:
        I[start:end] = amp  # absolute value (e.g., inhibition-induced: 80 → 70)
    else:
        I[start:end] = baseline + amp  # delta above baseline

    if ou_noise:
        noise_amp = abs(amp - baseline) if cell_type in _STEP_ABSOLUTE else abs(amp)
        I[burn_in_samples:] += _ou_noise(n_samples, dt, tau=ou_tau,
                                         sigma=ou_sigma_fraction * noise_amp)

    v, u, spikes_bin, cid = simulate_izhikevich(
        cell_type, I, dt, jitter=0, plotFlag=0, saveFlag=0, fid=''
    )

    # Strip burn-in
    spike_times = np.where(spikes_bin[burn_in_samples:])[0] * dt
    I_trial = I[burn_in_samples:]
    return I_trial, spike_times, dt, trial_duration_ms


def _bin_trial(I: np.ndarray, spike_times: np.ndarray, dt: float,
               bin_size_ms: float, trial_duration_ms: float
               ) -> Tuple[np.ndarray, np.ndarray]:
    """Bin stimulus and spikes to RNN resolution."""
    spb = int(bin_size_ms / dt)
    n_bins = int(trial_duration_ms / bin_size_ms)
    n_samples = n_bins * spb

    I_binned = I[:n_samples].reshape(n_bins, spb).mean(axis=1)

    edges = np.arange(0, trial_duration_ms + bin_size_ms, bin_size_ms)
    y_binned, _ = np.histogram(spike_times, bins=edges)
    return I_binned, y_binned[:n_bins].astype(np.float64)


def generate_training_data(
    cell_type: int,
    n_trials: int = 500,
    bin_size_ms: float = 1.0,
    seed: Optional[int] = None,
    verbose: bool = True,
    ou_noise: bool = True,
    ou_tau: float = 1.0,
    ou_sigma_fraction: float = 0.01,
    diverse_timing: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate trial-based RNN training data from an Izhikevich neuron.

    Args:
        cell_type: Izhikevich neuron type (1-21, excluding 10, 17)
        n_trials: number of trials
        bin_size_ms: time bin for RNN inputs/targets
        seed: random seed
        verbose: print progress
        diverse_timing: sample pre/step from log-normal/exponential distributions

    Returns:
        X:       (n_trials, max_bins, 1) binned current input (zero-padded)
        y:       (n_trials, max_bins)    binned spike counts (zero-padded)
        lengths: (n_trials,)             actual number of bins per trial
    """
    if seed is not None:
        np.random.seed(seed)

    Xs, ys = [], []
    for i in range(n_trials):
        I, st, dt, tdur = _simulate_trial(cell_type, diverse_timing=diverse_timing,
                                          ou_noise=ou_noise, ou_tau=ou_tau,
                                          ou_sigma_fraction=ou_sigma_fraction)
        Ib, yb = _bin_trial(I, st, dt, bin_size_ms, tdur)
        Xs.append(Ib)
        ys.append(yb)

    lengths = np.array([len(x) for x in Xs], dtype=np.int64)
    max_bins = int(lengths.max())
    X = np.zeros((n_trials, max_bins, 1))
    y = np.zeros((n_trials, max_bins))
    for i, (xb, yb) in enumerate(zip(Xs, ys)):
        X[i, :len(xb), 0] = xb
        y[i, :len(yb)] = yb

    if verbose:
        print(f"  Data: X {X.shape}, y {y.shape} | "
              f"total spikes={y.sum():.0f}, mean/trial={y.sum()/n_trials:.1f} | "
              f"lengths: min={lengths.min()}, max={lengths.max()}, mean={lengths.mean():.0f}")
    return X, y, lengths


def generate_test_trial(
    cell_type: int,
    bin_size_ms: float = 1.0,
    ou_tau: float = 1.0,
    ou_sigma_fraction: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a single diverse test trial (same distribution as training).
    Returns I_binned (seq_len,) and y_binned (seq_len,).
    """
    I, st, dt, tdur = _simulate_trial(cell_type, diverse_timing=True,
                                      ou_noise=True, ou_tau=ou_tau,
                                      ou_sigma_fraction=ou_sigma_fraction)
    return _bin_trial(I, st, dt, bin_size_ms, tdur)


def generate_clean_test_trial(
    cell_type: int,
    bin_size_ms: float = 1.0,
    pre_ms: float = 50.0,
    step_ms: float = 250.0,
    post_ms: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a single long, clean test trial (no noise, no jitter).
    Returns I_binned (seq_len,) and y_binned (seq_len,).
    """
    trial_ms = pre_ms + step_ms + post_ms
    I, st, dt, tdur = _simulate_trial(
        cell_type, trial_duration_ms=trial_ms,
        silence_pre_ms=pre_ms, step_duration_ms=step_ms,
        jitter=False, diverse_timing=False, ou_noise=False,
    )
    return _bin_trial(I, st, dt, bin_size_ms, tdur)


def generate_test_sequence(
    cell_type: int,
    T_ms: float = 1000.0,
    bin_size_ms: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Generate a long test sequence using the standard Izhikevich stimulus pattern.

    Returns:
        I_raw, spikes_raw, I_binned, y_binned, dt
    """
    I, dt = generate_izhikevich_stim(cell_type, T_ms)
    if I is None:
        raise ValueError(f"Unavailable cell type: {cell_type}")

    v, u, spikes, cid = simulate_izhikevich(
        cell_type, I, dt, jitter=0, plotFlag=0, saveFlag=0, fid=''
    )

    spb = int(bin_size_ms / dt)
    n_bins = len(I) // spb
    n_samples = n_bins * spb

    I_binned = I[:n_samples].reshape(n_bins, spb).mean(axis=1)
    spike_times = np.where(spikes[:n_samples])[0] * dt
    edges = np.arange(0, n_bins * bin_size_ms + bin_size_ms, bin_size_ms)
    y_binned, _ = np.histogram(spike_times, bins=edges)
    y_binned = y_binned[:n_bins].astype(np.float64)

    return I, spikes, I_binned, y_binned, dt
