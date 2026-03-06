"""
models.py - RNN architectures for fitting Izhikevich neuron dynamics.

Four models, all with 1 layer, 64 hidden units, tanh activation:
1. VanillaRNN         - Standard Elman RNN (nn.RNN baseline)
2. LearnableTauCTRNN  - CTRNN with learnable time constants via sigmoid reparameterization
3. FixedSpectrumCTRNN - CTRNN with fixed log-normal timescale distribution
4. PartitionedCTRNN   - CTRNN with explicit fast/slow compartments

All CTRNN variants implement the Euler-discretized leaky integrator:
    h[t] = (1 - alpha) * h[t-1] + alpha * (g * W_rec @ tanh(h[t-1]) + W_in @ x[t] + b)
where alpha = dt / tau controls each unit's integration rate.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# =============================================================================
# Base mixin (shared readout + learnable initial hidden state)
# =============================================================================

class _BaseRNNMixin:
    """Shared components: learnable h0, Linear -> Softplus readout with negative bias."""

    def _init_shared(self, hidden_size: int):
        self.h0 = nn.Parameter(torch.zeros(hidden_size))
        self.fc = nn.Linear(hidden_size, 1)
        nn.init.constant_(self.fc.bias, -3.0)
        self.softplus = nn.Softplus()

    def _get_h0(self, batch_size: int) -> torch.Tensor:
        return self.h0.unsqueeze(0).expand(batch_size, -1)

    def _readout(self, h_seq: torch.Tensor, return_sequence: bool) -> torch.Tensor:
        if return_sequence:
            return self.softplus(self.fc(h_seq)).squeeze(-1)
        else:
            return self.softplus(self.fc(h_seq[:, -1, :])).squeeze(-1)


# =============================================================================
# 1. Vanilla RNN (baseline)
# =============================================================================

class VanillaRNN(nn.Module, _BaseRNNMixin):
    """Standard Elman RNN with tanh nonlinearity, using nn.RNN."""

    def __init__(self, input_size: int = 1, hidden_size: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=1,
                          batch_first=True, nonlinearity='tanh')
        self._init_shared(hidden_size)

    def forward(self, x: torch.Tensor, return_sequence: bool = True,
                return_hidden: bool = False) -> torch.Tensor:
        h0_batch = self._get_h0(x.shape[0]).unsqueeze(0).contiguous()
        out, _ = self.rnn(x, h0_batch)
        readout = self._readout(out, return_sequence)
        if return_hidden:
            return readout, out
        return readout


# =============================================================================
# 2. Learnable Time Constant CTRNN
# =============================================================================

class LearnableTauCTRNN(nn.Module, _BaseRNNMixin):
    """
    CTRNN where each unit's time constant is learned via backpropagation.

    Reparameterization: tau = tau_min + sigmoid(rho) * (tau_max - tau_min)
    ensures tau in [tau_min, tau_max] and alpha = dt/tau in a stable range.
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 64,
                 dt: float = 1.0, tau_min: float = 1.0, tau_max: float = 200.0,
                 g: float = 1.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.dt = dt
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.g = g

        # Learnable unconstrained timescale parameter.
        # Init via logit of Uniform(eps, 1-eps) so taus start uniformly spread
        # across [tau_min, tau_max] rather than collapsed near the midpoint.
        eps = 0.05
        u = torch.empty(hidden_size).uniform_(eps, 1.0 - eps)
        self.rho = nn.Parameter(torch.log(u / (1.0 - u)))

        self.W_in = nn.Linear(input_size, hidden_size, bias=False)
        self.W_rec = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self._init_shared(hidden_size)
        self._init_weights()

    def _init_weights(self):
        # Initialize with standard deviation 1/sqrt(N)
        nn.init.normal_(self.W_rec.weight, 0, 1.0 / math.sqrt(self.hidden_size))
        # ... and keep the self.g multiplier in the forward pass.
        nn.init.normal_(self.W_in.weight, 0, 1.0 / math.sqrt(self.hidden_size))

    def get_alpha(self) -> torch.Tensor:

        tau = self.tau_min + torch.sigmoid(self.rho) * (self.tau_max - self.tau_min)
        return self.dt / tau

    def forward(self, x: torch.Tensor, return_sequence: bool = True,
                return_hidden: bool = False) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        h = self._get_h0(batch_size)
        alpha = self.get_alpha()
        h_seq = []
        for t in range(seq_len):
            drive = self.g * self.W_rec(torch.tanh(h)) + self.W_in(x[:, t, :]) + self.bias
            h = (1 - alpha) * h + alpha * drive
            h_seq.append(h)

        h_seq_tensor = torch.stack(h_seq, dim=1)
        readout = self._readout(h_seq_tensor, return_sequence)
        if return_hidden:
            return readout, h_seq_tensor
        return readout


# =============================================================================
# 3. Fixed Log-Normal Spectrum CTRNN
# =============================================================================

class FixedSpectrumCTRNN(nn.Module, _BaseRNNMixin):
    """
    CTRNN with a fixed log-normal distribution of time constants.

    The network learns to route signals through units with appropriate
    timescales rather than adapting the timescales themselves.
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 64,
                 dt: float = 1.0, tau_mean: float = 10.0, tau_std: float = 5.0,
                 g: float = 1.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.dt = dt
        self.g = g

        # Compute log-normal parameters from desired mean/std
        var = tau_std ** 2
        mu = math.log(tau_mean ** 2 / math.sqrt(var + tau_mean ** 2))
        sigma = math.sqrt(math.log(1 + var / tau_mean ** 2))

        # Sample tau, compute alpha, register as non-learnable buffer
        tau = torch.empty(hidden_size).log_normal_(mean=mu, std=sigma)
        alpha = torch.clamp(dt / tau, min=1e-4, max=1.0)
        self.register_buffer('alpha', alpha)

        self.W_in = nn.Linear(input_size, hidden_size, bias=False)
        self.W_rec = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self._init_shared(hidden_size)
        self._init_weights()

    def _init_weights(self):
        # Initialize with standard deviation 1/sqrt(N)
        nn.init.normal_(self.W_rec.weight, 0, 1.0 / math.sqrt(self.hidden_size))
        # ... and keep the self.g multiplier in the forward pass.
        nn.init.normal_(self.W_in.weight, 0, 1.0 / math.sqrt(self.hidden_size))

    def forward(self, x: torch.Tensor, return_sequence: bool = True,
                return_hidden: bool = False) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        h = self._get_h0(batch_size)
        h_seq = []
        for t in range(seq_len):
            drive = self.g * self.W_rec(torch.tanh(h)) + self.W_in(x[:, t, :]) + self.bias
            h = (1 - self.alpha) * h + self.alpha * drive
            h_seq.append(h)

        h_seq_tensor = torch.stack(h_seq, dim=1)
        readout = self._readout(h_seq_tensor, return_sequence)
        if return_hidden:
            return readout, h_seq_tensor
        return readout


# =============================================================================
# 4. Partitioned (Fast/Slow) CTRNN
# =============================================================================

class PartitionedCTRNN(nn.Module, _BaseRNNMixin):
    """
    CTRNN with explicit fast and slow compartments.

    The hidden state is split into two pools with distinct, fixed time constants,
    mimicking coupled fast/slow dynamical systems (e.g., voltage + recovery).
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 64,
                 dt: float = 1.0, tau_fast: float = 2.0, tau_slow: float = 50.0,
                 fast_ratio: float = 0.5, g: float = 1.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.dt = dt
        self.g = g
        self.n_fast = int(hidden_size * fast_ratio)
        self.n_slow = hidden_size - self.n_fast

        # Build alpha vector: [fast_units | slow_units]
        alpha = torch.cat([
            torch.full((self.n_fast,), dt / tau_fast),
            torch.full((self.n_slow,), dt / tau_slow),
        ])
        alpha = torch.clamp(alpha, min=1e-4, max=1.0)
        self.register_buffer('alpha', alpha)

        self.W_in = nn.Linear(input_size, hidden_size, bias=False)
        self.W_rec = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self._init_shared(hidden_size)
        self._init_weights()

    def _init_weights(self):
        # Initialize with standard deviation 1/sqrt(N)
        nn.init.normal_(self.W_rec.weight, 0, 1.0 / math.sqrt(self.hidden_size))
        # ... and keep the self.g multiplier in the forward pass.
        nn.init.normal_(self.W_in.weight, 0, 1.0 / math.sqrt(self.hidden_size))

    def forward(self, x: torch.Tensor, return_sequence: bool = True,
                return_hidden: bool = False) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        h = self._get_h0(batch_size)

        h_seq = []
        for t in range(seq_len):
            drive = self.g * self.W_rec(torch.tanh(h)) + self.W_in(x[:, t, :]) + self.bias
            h = (1 - self.alpha) * h + self.alpha * drive
            h_seq.append(h)

        h_seq_tensor = torch.stack(h_seq, dim=1)
        readout = self._readout(h_seq_tensor, return_sequence)
        if return_hidden:
            return readout, h_seq_tensor
        return readout


# =============================================================================
# 5. Multi-Partition CTRNN (generalised N-compartment version)
# =============================================================================

class MultiPartitionedCTRNN(nn.Module, _BaseRNNMixin):
    """
    CTRNN with N equally-sized compartments, each assigned a fixed tau.

    A generalisation of PartitionedCTRNN to arbitrary numbers of timescales.
    Units are divided as evenly as possible; any remainder goes to the last group.
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 64,
                 dt: float = 1.0,
                 taus: tuple = (2.0, 25.0, 50.0, 75.0, 100.0),
                 g: float = 1.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.dt = dt
        self.g = g
        self.taus = taus

        # Divide units as evenly as possible across compartments
        n = len(taus)
        base, remainder = divmod(hidden_size, n)
        sizes = [base + 1 if i < remainder else base for i in range(n)]
        # For 64 units and 5 taus, sizes becomes: [13, 13, 13, 13, 12]
        self.sizes = sizes

        alpha = torch.cat([
            torch.full((sz,), dt / tau)
            for sz, tau in zip(sizes, taus)
        ])
        alpha = torch.clamp(alpha, min=1e-4, max=1.0)
        self.register_buffer('alpha', alpha)

        self.W_in = nn.Linear(input_size, hidden_size, bias=False)
        self.W_rec = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self._init_shared(hidden_size)
        self._init_weights()

    def _init_weights(self):
        # Initialize with standard deviation 1/sqrt(N)
        nn.init.normal_(self.W_rec.weight, 0, 1.0 / math.sqrt(self.hidden_size))
        # ... and keep the self.g multiplier in the forward pass.
        nn.init.normal_(self.W_in.weight, 0, 1.0 / math.sqrt(self.hidden_size))

    def forward(self, x: torch.Tensor, return_sequence: bool = True,
                return_hidden: bool = False) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        h = self._get_h0(batch_size)

        h_seq = []
        for t in range(seq_len):
            drive = self.g * self.W_rec(torch.tanh(h)) + self.W_in(x[:, t, :]) + self.bias
            h = (1 - self.alpha) * h + self.alpha * drive
            h_seq.append(h)

        h_seq_tensor = torch.stack(h_seq, dim=1)
        readout = self._readout(h_seq_tensor, return_sequence)
        if return_hidden:
            return readout, h_seq_tensor
        return readout


# =============================================================================
# 6. Learnable Partitioned (Fast/Slow) CTRNN
# =============================================================================

class LearnablePartitionedCTRNN(nn.Module, _BaseRNNMixin):
    """
    CTRNN with two learnable timescale parameters: one for the fast compartment,
    one for the slow compartment. Combines the structural prior of PartitionedCTRNN
    with backpropagation flexibility.
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 64,
                 dt: float = 1.0, tau_min: float = 1.0, tau_max: float = 200.0,
                 init_tau_fast: float = 2.0, init_tau_slow: float = 50.0,
                 fast_ratio: float = 0.5, g: float = 1.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.dt = dt
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.g = g
        self.n_fast = int(hidden_size * fast_ratio)
        self.n_slow = hidden_size - self.n_fast

        # Initialise rho via logit so that sigmoid(rho) maps to the target tau
        def _tau_to_rho(tau):
            p = (tau - tau_min) / (tau_max - tau_min)
            return math.log(p / (1 - p))

        self.rho_fast = nn.Parameter(torch.tensor(_tau_to_rho(init_tau_fast)))
        self.rho_slow = nn.Parameter(torch.tensor(_tau_to_rho(init_tau_slow)))

        self.W_in = nn.Linear(input_size, hidden_size, bias=False)
        self.W_rec = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self._init_shared(hidden_size)
        self._init_weights()

    def _init_weights(self):
        # Initialize with standard deviation 1/sqrt(N)
        nn.init.normal_(self.W_rec.weight, 0, 1.0 / math.sqrt(self.hidden_size))
        # ... and keep the self.g multiplier in the forward pass.
        nn.init.normal_(self.W_in.weight, 0, 1.0 / math.sqrt(self.hidden_size))

    def get_alpha(self) -> torch.Tensor:
        tau_fast = self.tau_min + torch.sigmoid(self.rho_fast) * (self.tau_max - self.tau_min)
        tau_slow = self.tau_min + torch.sigmoid(self.rho_slow) * (self.tau_max - self.tau_min)
        alpha_fast = (self.dt / tau_fast).expand(self.n_fast)
        alpha_slow = (self.dt / tau_slow).expand(self.n_slow)
        return torch.cat([alpha_fast, alpha_slow])

    def forward(self, x: torch.Tensor, return_sequence: bool = True,
                return_hidden: bool = False) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        h = self._get_h0(batch_size)
        alpha = self.get_alpha()

        h_seq = []
        for t in range(seq_len):
            drive = self.g * self.W_rec(torch.tanh(h)) + self.W_in(x[:, t, :]) + self.bias
            h = (1 - alpha) * h + alpha * drive
            h_seq.append(h)

        h_seq_tensor = torch.stack(h_seq, dim=1)
        readout = self._readout(h_seq_tensor, return_sequence)
        if return_hidden:
            return readout, h_seq_tensor
        return readout


# =============================================================================
# 7. Clockwork RNN (Koutnik et al. 2014)
# =============================================================================

class ClockworkRNN(nn.Module, _BaseRNNMixin):
    """
    Discrete-time Clockwork RNN with tanh/g voltage-based formulation.

    Modules are ordered fastest-to-slowest. The block-upper-triangular
    structural mask enforces that faster modules receive input only from
    themselves and slower modules (information flows slow-to-fast).
    At each timestep t, only units whose period evenly divides t are updated;
    the rest retain their previous state.
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 64,
                 periods: tuple = (1, 2, 4, 8, 16, 32, 64, 128),
                 g: float = 1.5, **kwargs):
        super().__init__()
        n_modules = len(periods)
        assert hidden_size % n_modules == 0, (
            f"hidden_size ({hidden_size}) must be divisible by n_modules ({n_modules})"
        )
        self.hidden_size = hidden_size
        self.g = g
        units_per_module = hidden_size // n_modules

        # Map each hidden unit to its module's period (fastest to slowest)
        periods_tensor = torch.cat([
            torch.full((units_per_module,), p, dtype=torch.long)
            for p in periods
        ])
        self.register_buffer('unit_periods', periods_tensor)

        # Block-upper-triangular structural mask:
        # W_mask[i, j] = 1 if unit i can receive from unit j (period[i] <= period[j])
        pi = periods_tensor.unsqueeze(1)   # (hidden_size, 1)
        pj = periods_tensor.unsqueeze(0)   # (1, hidden_size)
        W_mask = (pi <= pj).float()
        self.register_buffer('W_mask', W_mask)

        self.W_in = nn.Linear(input_size, hidden_size, bias=False)
        self.W_rec = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self._init_shared(hidden_size)
        self._init_weights()

    def _init_weights(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        nn.init.normal_(self.W_in.weight, 0, std)
        nn.init.normal_(self.W_rec.weight, 0, std)

    def forward(self, x: torch.Tensor, return_sequence: bool = True,
                return_hidden: bool = False) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        h = self._get_h0(batch_size)
        h_seq = []

        for t in range(seq_len):
            # Units whose period evenly divides t are active at this step
            active_mask = (t % self.unit_periods == 0).unsqueeze(0)  # (1, hidden_size)

            # Proposed new state for all units (structural mask on W_rec)
            h_new = (self.g * F.linear(torch.tanh(h), self.W_rec.weight * self.W_mask)
                     + self.W_in(x[:, t, :]) + self.bias)

            # Only active units update; the rest keep their previous state
            h = torch.where(active_mask, h_new, h)
            h_seq.append(h)

        h_seq_tensor = torch.stack(h_seq, dim=1)
        readout = self._readout(h_seq_tensor, return_sequence)
        if return_hidden:
            return readout, h_seq_tensor
        return readout


# =============================================================================
# Model registry for convenient access
# =============================================================================

MODEL_REGISTRY = {
    'VanillaRNN': VanillaRNN,
    'LearnableTau': LearnableTauCTRNN,
    'FixedSpectrum': FixedSpectrumCTRNN,
    'Partitioned': PartitionedCTRNN,
    'MultiPartitioned': MultiPartitionedCTRNN,
    'LearnablePartitioned': LearnablePartitionedCTRNN,
    'Clockwork': ClockworkRNN,
}

MODEL_COLORS = {
    'VanillaRNN': 'tab:blue',
    'LearnableTau': 'tab:orange',
    'FixedSpectrum': 'tab:green',
    'Partitioned': 'tab:red',
    'MultiPartitioned': 'tab:purple',
    'LearnablePartitioned': 'tab:brown',
    'Clockwork': 'tab:cyan',
}


def build_model(name: str, input_size: int = 1, hidden_size: int = 64,
                **kwargs) -> nn.Module:
    """Instantiate a model by name from the registry.
    dt is silently dropped for VanillaRNN which has no continuous-time dynamics.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODEL_REGISTRY.keys())}")
    if name in ('VanillaRNN', 'Clockwork'):
        kwargs.pop('dt', None)
    return MODEL_REGISTRY[name](input_size=input_size, hidden_size=hidden_size, **kwargs)
