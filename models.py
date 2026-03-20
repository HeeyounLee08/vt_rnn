"""
models.py - RNN architectures for fitting Izhikevich neuron dynamics.

Models (all with configurable activation: tanh or relu):
1. VanillaRNN               - Standard Elman RNN
2. LearnableTauCTRNN        - CTRNN with learnable time constants
3. FixedSpectrumCTRNN       - CTRNN with fixed log-normal timescale distribution
4. PartitionedCTRNN         - CTRNN with explicit fast/slow compartments
5. MultiPartitionedCTRNN    - CTRNN with N equally-sized compartments
6. LearnablePartitionedCTRNN - CTRNN with learnable fast/slow partitions
7. ClockworkRNN             - Clockwork RNN (Koutnik et al. 2014)
8. PLRNN                    - Piecewise-Linear RNN (Durstewitz 2017)
9. PureLogTauCTRNN          - CTRNN with log-space tau parameterisation
10. PureLogPartitionedCTRNN  - CTRNN with log-space fast/slow partition

Two hidden-state views (selectable via `view` parameter):
  membrane_potential: nonlinearity on recurrent input only; state integrates linearly
  firing_rate:        nonlinearity wraps the full update; state is always post-activation

Clockwork and PLRNN have fixed membrane_potential semantics and ignore the view flag.
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

    def _init_activation(self, activation_name: str):
        self.activation_name = activation_name.lower()
        if self.activation_name == 'relu':
            self.act_fn = torch.relu
            self.default_g = 0.5   # lower gain for unbounded ReLU
        elif self.activation_name == 'tanh':
            self.act_fn = torch.tanh
            self.default_g = 1.5   # higher gain for squashed tanh
        else:
            raise ValueError(f"Unsupported activation: {activation_name}")

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

    def _ctrnn_step(self, h, x_t, alpha):
        """Single CTRNN Euler step, dispatched by self.view. State clamped to [-50, 50]."""
        if self.view == 'firing_rate':
            pre = self.g * self.W_rec(h) + self.W_in(x_t) + self.bias
            h = self.act_fn((1 - alpha) * h + alpha * pre)
        else:
            # membrane_potential (default)
            drive = self.g * self.W_rec(self.act_fn(h)) + self.W_in(x_t) + self.bias
            h = (1 - alpha) * h + alpha * drive
        return torch.clamp(h, min=-50.0, max=50.0)


# =============================================================================
# 1. Vanilla RNN (baseline)
# =============================================================================

class VanillaRNN(nn.Module, _BaseRNNMixin):
    """
    Standard Elman RNN with configurable view.
      membrane_potential: h = W_in x + W_rec act(h) + b
      firing_rate:        h = act(W_in x + W_rec h + b)
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 64,
                 activation: str = 'tanh', view: str = 'membrane_potential', **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.view = view
        self._init_activation(activation)
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
            if self.view == 'firing_rate':
                h = self.act_fn(self.W_in(x[:, t, :]) + self.W_rec(h) + self.bias)
            else:
                h = self.W_in(x[:, t, :]) + self.W_rec(self.act_fn(h)) + self.bias
            h = torch.clamp(h, min=-50.0, max=50.0)
            h_seq.append(h)
        h_seq_tensor = torch.stack(h_seq, dim=1)
        readout = self._readout(h_seq_tensor, return_sequence)
        if return_hidden:
            return readout, h_seq_tensor
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
                 g: float = None, activation: str = 'tanh',
                 view: str = 'membrane_potential'):
        super().__init__()
        self.hidden_size = hidden_size
        self.dt = dt
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.view = view
        self._init_activation(activation)
        self.g = g if g is not None else self.default_g

        eps = 0.05
        u = torch.empty(hidden_size).uniform_(eps, 1.0 - eps)
        self.rho = nn.Parameter(torch.log(u / (1.0 - u)))

        self.W_in = nn.Linear(input_size, hidden_size, bias=False)
        self.W_rec = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self._init_shared(hidden_size)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.W_rec.weight, 0, 1.0 / math.sqrt(self.hidden_size))
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
            h = self._ctrnn_step(h, x[:, t, :], alpha)
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
                 g: float = None, activation: str = 'tanh',
                 view: str = 'membrane_potential'):
        super().__init__()
        self.hidden_size = hidden_size
        self.dt = dt
        self.view = view
        self._init_activation(activation)
        self.g = g if g is not None else self.default_g

        var = tau_std ** 2
        mu = math.log(tau_mean ** 2 / math.sqrt(var + tau_mean ** 2))
        sigma = math.sqrt(math.log(1 + var / tau_mean ** 2))

        tau = torch.empty(hidden_size).log_normal_(mean=mu, std=sigma)
        alpha = torch.clamp(dt / tau, min=1e-4, max=1.0)
        self.register_buffer('alpha', alpha)

        self.W_in = nn.Linear(input_size, hidden_size, bias=False)
        self.W_rec = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self._init_shared(hidden_size)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.W_rec.weight, 0, 1.0 / math.sqrt(self.hidden_size))
        nn.init.normal_(self.W_in.weight, 0, 1.0 / math.sqrt(self.hidden_size))

    def forward(self, x: torch.Tensor, return_sequence: bool = True,
                return_hidden: bool = False) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        h = self._get_h0(batch_size)
        h_seq = []
        for t in range(seq_len):
            h = self._ctrnn_step(h, x[:, t, :], self.alpha)
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
                 fast_ratio: float = 0.5, g: float = None, activation: str = 'tanh',
                 view: str = 'membrane_potential'):
        super().__init__()
        self.hidden_size = hidden_size
        self.dt = dt
        self.n_fast = int(hidden_size * fast_ratio)
        self.n_slow = hidden_size - self.n_fast
        self.view = view
        self._init_activation(activation)
        self.g = g if g is not None else self.default_g

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
        nn.init.normal_(self.W_rec.weight, 0, 1.0 / math.sqrt(self.hidden_size))
        nn.init.normal_(self.W_in.weight, 0, 1.0 / math.sqrt(self.hidden_size))

    def forward(self, x: torch.Tensor, return_sequence: bool = True,
                return_hidden: bool = False) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        h = self._get_h0(batch_size)

        h_seq = []
        for t in range(seq_len):
            h = self._ctrnn_step(h, x[:, t, :], self.alpha)
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
                 g: float = None, activation: str = 'tanh',
                 view: str = 'membrane_potential'):
        super().__init__()
        self.hidden_size = hidden_size
        self.dt = dt
        self.taus = taus
        self.view = view
        self._init_activation(activation)
        self.g = g if g is not None else self.default_g

        n = len(taus)
        base, remainder = divmod(hidden_size, n)
        sizes = [base + 1 if i < remainder else base for i in range(n)]
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
        nn.init.normal_(self.W_rec.weight, 0, 1.0 / math.sqrt(self.hidden_size))
        nn.init.normal_(self.W_in.weight, 0, 1.0 / math.sqrt(self.hidden_size))

    def forward(self, x: torch.Tensor, return_sequence: bool = True,
                return_hidden: bool = False) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        h = self._get_h0(batch_size)

        h_seq = []
        for t in range(seq_len):
            h = self._ctrnn_step(h, x[:, t, :], self.alpha)
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
                 fast_ratio: float = 0.5, g: float = None, activation: str = 'tanh',
                 view: str = 'membrane_potential'):
        super().__init__()
        self.hidden_size = hidden_size
        self.dt = dt
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.n_fast = int(hidden_size * fast_ratio)
        self.n_slow = hidden_size - self.n_fast
        self.view = view
        self._init_activation(activation)
        self.g = g if g is not None else self.default_g

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
        nn.init.normal_(self.W_rec.weight, 0, 1.0 / math.sqrt(self.hidden_size))
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
            h = self._ctrnn_step(h, x[:, t, :], alpha)
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
    Discrete-time Clockwork RNN with configurable activation.

    Modules are ordered fastest-to-slowest. The block-upper-triangular
    structural mask enforces that faster modules receive input only from
    themselves and slower modules (information flows slow-to-fast).
    At each timestep t, only units whose period evenly divides t are updated;
    the rest retain their previous state.

    Always uses membrane_potential semantics; the view flag is ignored.
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 64,
                 periods: tuple = (1, 2, 4, 8, 16, 32, 64, 128),
                 g: float = None, activation: str = 'tanh', **kwargs):
        super().__init__()
        periods = list(periods)
        while len(periods) > 1 and hidden_size % len(periods) != 0:
            periods = periods[:-1]
        n_modules = len(periods)
        self.hidden_size = hidden_size
        self._init_activation(activation)
        self.g = g if g is not None else self.default_g
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
            active_mask = (t % self.unit_periods == 0).unsqueeze(0)
            h_new = (self.g * F.linear(self.act_fn(h), self.W_rec.weight * self.W_mask)
                     + self.W_in(x[:, t, :]) + self.bias)
            h = torch.where(active_mask, h_new, h)
            h = torch.clamp(h, min=-50.0, max=50.0)
            h_seq.append(h)

        h_seq_tensor = torch.stack(h_seq, dim=1)
        readout = self._readout(h_seq_tensor, return_sequence)
        if return_hidden:
            return readout, h_seq_tensor
        return readout


# =============================================================================
# 8. Piecewise-Linear RNN (Durstewitz 2017)
# =============================================================================

class PLRNN(nn.Module, _BaseRNNMixin):
    """
    Piecewise-Linear RNN (Durstewitz, 2017).
    z_t = A * z_{t-1} + W * act_fn(z_{t-1} - theta) + W_in * x_t + bias
    A is a diagonal parameter (auto-regressive time constants).
    W is a strictly off-diagonal recurrent weight matrix.

    Always uses membrane_potential semantics; the view flag is ignored.
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 64,
                 activation: str = 'relu', **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self._init_activation(activation)

        # Diagonal auto-regressive weights (initialized near 1 for slow integration)
        self.A = nn.Parameter(torch.empty(hidden_size).uniform_(0.5, 0.99))

        # Off-diagonal recurrent weights
        self.W_rec = nn.Linear(hidden_size, hidden_size, bias=False)
        with torch.no_grad():
            self.W_rec.weight.fill_diagonal_(0.0)
        self.register_buffer('W_mask', 1.0 - torch.eye(hidden_size))

        self.W_in = nn.Linear(input_size, hidden_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.theta = nn.Parameter(torch.zeros(hidden_size))

        self._init_shared(hidden_size)
        self._init_weights()

    def _init_weights(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        nn.init.normal_(self.W_in.weight, 0, std)
        nn.init.normal_(self.W_rec.weight, 0, std)

    def forward(self, x: torch.Tensor, return_sequence: bool = True,
                return_hidden: bool = False) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        z = self._get_h0(batch_size)

        A_stable = self.A.clamp(0.0, 0.99)
        z_seq = []
        for t in range(seq_len):
            phi_z = self.act_fn(z - self.theta)
            lateral = F.linear(phi_z, self.W_rec.weight * self.W_mask)
            z = A_stable * z + lateral + self.W_in(x[:, t, :]) + self.bias
            z = torch.clamp(z, min=-50.0, max=50.0)
            z_seq.append(z)

        z_seq_tensor = torch.stack(z_seq, dim=1)
        readout = self._readout(z_seq_tensor, return_sequence)
        if return_hidden:
            return readout, z_seq_tensor
        return readout


# =============================================================================
# 9. Pure Log-Space Tau CTRNN
# =============================================================================

class PureLogTauCTRNN(nn.Module, _BaseRNNMixin):
    """
    CTRNN with log-space tau parameterisation: tau = exp(T).

    Unlike sigmoid-based reparameterisation (LearnableTauCTRNN), the log-space
    approach avoids vanishing gradients at extreme timescales, allowing the
    network to learn very fast or very slow dynamics without hitting saturation.
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 64,
                 dt: float = 1.0, init_tau_min: float = 1.0, init_tau_max: float = 200.0,
                 g: float = None, activation: str = 'tanh',
                 view: str = 'membrane_potential'):
        super().__init__()
        self.hidden_size = hidden_size
        self.dt = dt
        self.view = view
        self._init_activation(activation)
        self.g = g if g is not None else self.default_g

        # T = log(tau), so tau = exp(T). Init uniform in tau-space.
        init_taus = torch.empty(hidden_size).uniform_(init_tau_min, init_tau_max)
        self.T = nn.Parameter(torch.log(init_taus))

        self.W_in = nn.Linear(input_size, hidden_size, bias=False)
        self.W_rec = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self._init_shared(hidden_size)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.W_rec.weight, 0, 1.0 / math.sqrt(self.hidden_size))
        nn.init.normal_(self.W_in.weight, 0, 1.0 / math.sqrt(self.hidden_size))

    def get_alpha(self) -> torch.Tensor:
        tau = torch.exp(torch.clamp(self.T, min=-10, max=10))
        return torch.clamp(self.dt / tau, max=1.0)

    def forward(self, x: torch.Tensor, return_sequence: bool = True,
                return_hidden: bool = False) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        h = self._get_h0(batch_size)
        alpha = self.get_alpha()
        h_seq = []
        for t in range(seq_len):
            h = self._ctrnn_step(h, x[:, t, :], alpha)
            h_seq.append(h)

        h_seq_tensor = torch.stack(h_seq, dim=1)
        readout = self._readout(h_seq_tensor, return_sequence)
        if return_hidden:
            return readout, h_seq_tensor
        return readout


# =============================================================================
# 10. Pure Log-Space Partitioned CTRNN
# =============================================================================

class PureLogPartitionedCTRNN(nn.Module, _BaseRNNMixin):
    """
    CTRNN with two log-space tau parameters: one for fast, one for slow compartment.

    Combines the structural fast/slow prior of PartitionedCTRNN with
    the gradient-friendly log-space parameterisation of PureLogTauCTRNN.
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 64,
                 dt: float = 1.0, init_tau_fast: float = 2.0, init_tau_slow: float = 50.0,
                 fast_ratio: float = 0.5, g: float = None, activation: str = 'tanh',
                 view: str = 'membrane_potential'):
        super().__init__()
        self.hidden_size = hidden_size
        self.dt = dt
        self.n_fast = int(hidden_size * fast_ratio)
        self.n_slow = hidden_size - self.n_fast
        self.view = view
        self._init_activation(activation)
        self.g = g if g is not None else self.default_g

        self.T_fast = nn.Parameter(torch.log(torch.tensor(float(init_tau_fast))))
        self.T_slow = nn.Parameter(torch.log(torch.tensor(float(init_tau_slow))))

        self.W_in = nn.Linear(input_size, hidden_size, bias=False)
        self.W_rec = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self._init_shared(hidden_size)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.W_rec.weight, 0, 1.0 / math.sqrt(self.hidden_size))
        nn.init.normal_(self.W_in.weight, 0, 1.0 / math.sqrt(self.hidden_size))

    def get_alpha(self) -> torch.Tensor:
        tau_fast = torch.exp(torch.clamp(self.T_fast, min=-10, max=10))
        tau_slow = torch.exp(torch.clamp(self.T_slow, min=-10, max=10))
        alpha_fast = torch.clamp(self.dt / tau_fast, max=1.0).expand(self.n_fast)
        alpha_slow = torch.clamp(self.dt / tau_slow, max=1.0).expand(self.n_slow)
        return torch.cat([alpha_fast, alpha_slow])

    def forward(self, x: torch.Tensor, return_sequence: bool = True,
                return_hidden: bool = False) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        h = self._get_h0(batch_size)
        alpha = self.get_alpha()
        h_seq = []
        for t in range(seq_len):
            h = self._ctrnn_step(h, x[:, t, :], alpha)
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
    'PLRNN': PLRNN,
    'PureLogTau': PureLogTauCTRNN,
    'PureLogPartitioned': PureLogPartitionedCTRNN,
}

# Models that only support membrane_potential view
MEMBRANE_ONLY_MODELS = {'Clockwork', 'PLRNN'}

MODEL_COLORS = {
    'VanillaRNN': 'tab:blue',
    'LearnableTau': 'tab:orange',
    'FixedSpectrum': 'tab:green',
    'Partitioned': 'tab:red',
    'MultiPartitioned': 'tab:purple',
    'LearnablePartitioned': 'tab:brown',
    'Clockwork': 'tab:cyan',
    'PLRNN': 'tab:pink',
    'PureLogTau': '#e6ab02',
    'PureLogPartitioned': '#66a61e',
}


def build_model(name: str, input_size: int = 1, hidden_size: int = 64,
                **kwargs) -> nn.Module:
    """Instantiate a model by name from the registry.
    dt is silently dropped for models without continuous-time dynamics.
    view is silently dropped for Clockwork and PLRNN.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODEL_REGISTRY.keys())}")
    if name in ('VanillaRNN', 'Clockwork', 'PLRNN'):
        kwargs.pop('dt', None)
    if name in MEMBRANE_ONLY_MODELS:
        kwargs.pop('view', None)
    return MODEL_REGISTRY[name](input_size=input_size, hidden_size=hidden_size, **kwargs)
