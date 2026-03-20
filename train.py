"""
train.py - Training loop for RNN models on Izhikevich spike data.

Provides a unified Trainer class that handles:
- Z-score normalization of inputs
- Poisson NLL loss (appropriate for spike count targets)
- Gradient clipping and LR scheduling
- Per-epoch loss tracking
"""

import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


class EarlyStopper:
    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, loss: float) -> bool:
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


class Trainer:
    """
    Train a single nn.Module on (X, y) spike data with Poisson NLL loss.

    Usage:
        trainer = Trainer(model, device='cuda')
        losses = trainer.fit(X_train, y_train, n_epochs=50)
        y_pred = trainer.predict(X_test)
    """

    def __init__(self, model: nn.Module, lr: float = 1e-3, batch_size: int = 64,
                 device: Optional[str] = None, tau_lr_multiplier: float = 1.0):
        if device is None:
            device = 'cpu'
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.lr = lr
        self.batch_size = batch_size
        self.tau_lr_multiplier = tau_lr_multiplier

        # Set during fit
        self.X_mean: Optional[np.ndarray] = None
        self.X_std: Optional[np.ndarray] = None
        self.train_losses: List[float] = []
        self.diverged: bool = False

    def _normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self.X_mean = X.mean(axis=(0, 1), keepdims=True)
            self.X_std = X.std(axis=(0, 1), keepdims=True)
        return (X - self.X_mean) / (self.X_std + 1e-8)

    def fit(self, X: np.ndarray, y: np.ndarray, n_epochs: int = 50,
            lengths: Optional[np.ndarray] = None,
            verbose: bool = True, use_freeze_schedule: bool = True,
            aux_variance_gamma: float = 0.0) -> List[float]:
        """
        Train the model.

        Args:
            X: (n_trials, seq_len, n_features)
            y: (n_trials, seq_len) spike counts
            n_epochs: training epochs
            lengths: (n_trials,) actual sequence lengths for loss masking (None = no masking)
            verbose: print progress
            use_freeze_schedule: if True, implement alternating freeze schedule for learnable timescales
            aux_variance_gamma: weight for auxiliary temporal variance penalty on hidden states.
                Penalises rapid oscillation: gamma * mean((h[:,1:] - h[:,:-1])^2).
                Set to 0.0 (default) to disable.

        Returns:
            List of per-epoch losses
        """
        # Detect if model has learnable timescales (rho for sigmoid, T/T_fast/T_slow for log-space)
        _tau_param_names = {'rho', 'rho_fast', 'rho_slow', 'T', 'T_fast', 'T_slow'}
        def _is_tau_param(name):
            # name is like 'rho', 'T_fast', or 'module.rho' — check the leaf
            leaf = name.rsplit('.', 1)[-1]
            return leaf in _tau_param_names
        has_learnable_tau = any(_is_tau_param(name) for name, _ in self.model.named_parameters())
        if not has_learnable_tau:
            use_freeze_schedule = False  # Don't freeze if no tau parameters exist

        X_norm = self._normalize(X, fit=True)

        X_t = torch.FloatTensor(X_norm).to(self.device)
        y_t = torch.FloatTensor(y.copy()).to(self.device)

        if lengths is not None:
            # Build boolean mask (n_trials, seq_len): True for valid timesteps
            seq_len = X.shape[1]
            idx = torch.arange(seq_len).unsqueeze(0)          # (1, seq_len)
            lens = torch.tensor(lengths).unsqueeze(1)          # (n_trials, 1)
            mask_t = (idx < lens).float().to(self.device)      # (n_trials, seq_len)
            loader = DataLoader(TensorDataset(X_t, y_t, mask_t),
                                batch_size=self.batch_size, shuffle=True)
        else:
            loader = DataLoader(TensorDataset(X_t, y_t),
                                batch_size=self.batch_size, shuffle=True)

        base_params, rho_params = [], []
        for pname, param in self.model.named_parameters():
            (rho_params if _is_tau_param(pname) else base_params).append(param)
        optimizer = torch.optim.Adam([
            {'params': base_params, 'lr': self.lr},
            {'params': rho_params, 'lr': self.lr * self.tau_lr_multiplier},
        ])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        self.train_losses = []
        self.model.train()
        early_stopper = EarlyStopper(patience=30, min_delta=1e-4)

        # Phase boundaries for freeze schedule
        phase1_end = int(n_epochs * 0.1)
        phase2_end = int(n_epochs * 0.6)

        for epoch in range(n_epochs):
            # Implement alternating freeze schedule for learnable timescales
            if use_freeze_schedule:
                if epoch < phase1_end:
                    # Phase 1: Freeze timescale params, train spatial weights
                    for name, param in self.model.named_parameters():
                        param.requires_grad = not _is_tau_param(name)
                elif epoch < phase2_end:
                    # Phase 2: Train everything
                    for name, param in self.model.named_parameters():
                        param.requires_grad = True
                else:
                    # Phase 3: Freeze spatial weights, train timescale params
                    for name, param in self.model.named_parameters():
                        param.requires_grad = _is_tau_param(name)

                # Log phase transitions
                if verbose:
                    if epoch == phase1_end:
                        print(f"    Epoch {epoch+1}: Unfreezing timescales (Phase 2)")
                    elif epoch == phase2_end:
                        print(f"    Epoch {epoch+1}: Freezing spatial weights (Phase 3)")

            use_aux = aux_variance_gamma > 0.0
            epoch_loss = 0.0
            batch_idx = 0
            for batch in loader:
                Xb, yb = batch[0], batch[1]
                optimizer.zero_grad()
                if use_aux:
                    y_pred, h_seq = self.model(Xb, return_sequence=True, return_hidden=True)
                else:
                    y_pred = self.model(Xb, return_sequence=True)
                loss_per_step = y_pred - yb * torch.log(y_pred + 1e-8)
                if lengths is not None:
                    mask_b = batch[2]
                    loss = (loss_per_step * mask_b).sum() / mask_b.sum()
                else:
                    loss = loss_per_step.mean()

                # Auxiliary temporal variance penalty
                if use_aux:
                    dh = h_seq[:, 1:, :] - h_seq[:, :-1, :]
                    loss = loss + aux_variance_gamma * (dh ** 2).mean()

                # Singular value regularisation: penalise effective spectral radius > threshold
                # Effective radius = g * sigma_max(W_rec), so threshold on W_rec = target / g
                # For large matrices (>=256), compute every 5 batches to save O(N³) cost
                lambda_stb = 0.01
                g = getattr(self.model, 'g', 1.0)
                sv_threshold = 1.5 / g  # target effective spectral radius ~1.5
                for pname, param in self.model.named_parameters():
                    if 'W_rec.weight' in pname:
                        svd_every = 5 if param.shape[0] >= 256 else 1
                        if batch_idx % svd_every == 0:
                            try:
                                max_sv = torch.linalg.svdvals(param)[0]
                                loss = loss + lambda_stb * F.relu(max_sv - sv_threshold)
                            except torch.linalg.LinAlgError:
                                pass  # ill-conditioned (e.g. tiny hidden size); skip penalty

                loss.backward()
                batch_idx += 1
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                self._clip_wrec_spectral_radius()
                epoch_loss += loss.item()

            avg = epoch_loss / len(loader)
            if not math.isfinite(avg):
                if verbose:
                    print(f"    *** NaN/Inf loss at epoch {epoch+1}, stopping training ***")
                self.diverged = True
                break
            self.train_losses.append(avg)
            scheduler.step(avg)

            if verbose and (epoch + 1) % max(1, n_epochs // 5) == 0:
                print(f"    Epoch {epoch+1:3d}/{n_epochs}  Loss: {avg:.4f}")

            # Early stopping — only allowed outside freeze schedule or once in Phase 3
            can_stop = (not use_freeze_schedule) or (epoch >= phase2_end)
            if can_stop and early_stopper(avg):
                if verbose:
                    print(f"    Early stopping triggered at epoch {epoch+1} "
                          f"(Best loss: {early_stopper.best_loss:.4f})")
                break

        return self.train_losses

    def _clip_wrec_spectral_radius(self):
        """After each optimizer step, rescale W_rec if effective spectral radius exceeds limit.
        Effective radius = g * sigma_max(W_rec). Hard limit at 2.5 to allow autonomous dynamics."""
        if not hasattr(self.model, 'W_rec'):
            return
        g = getattr(self.model, 'g', 1.0)
        max_radius = 2.5 / g  # effective hard limit of 2.5
        with torch.no_grad():
            W = self.model.W_rec.weight
            if not torch.isfinite(W).all():
                return
            try:
                sigma = torch.linalg.matrix_norm(W, ord=2)
            except torch.linalg.LinAlgError:
                return  # ill-conditioned; skip clipping
            if sigma > max_radius:
                self.model.W_rec.weight.mul_(max_radius / sigma)

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict firing rates for input sequences. Returns (n, seq_len)."""
        X_norm = self._normalize(X, fit=False)
        self.model.eval()
        X_t = torch.FloatTensor(X_norm).to(self.device)
        out = self.model(X_t, return_sequence=True).cpu().numpy()
        return np.clip(np.nan_to_num(out, nan=0.0, posinf=1e6), 0, 1e6)

    def save(self, path: Path):
        """Save trainer state (model weights + normalisation stats + losses) to a .pt file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'X_mean':      self.X_mean,
            'X_std':       self.X_std,
            'train_losses': self.train_losses,
            'lr':          self.lr,
            'batch_size':  self.batch_size,
            'tau_lr_multiplier': self.tau_lr_multiplier,
        }, path)

    @classmethod
    def load(cls, path: Path, model: nn.Module, device: Optional[str] = None) -> 'Trainer':
        """Load a saved trainer. The model architecture must already be constructed."""
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        trainer = cls(model,
                      lr=ckpt.get('lr', 1e-3),
                      batch_size=ckpt.get('batch_size', 64),
                      device=device,
                      tau_lr_multiplier=ckpt.get('tau_lr_multiplier', 1.0))
        trainer.model.load_state_dict(ckpt['model_state'])
        trainer.X_mean = ckpt['X_mean']
        trainer.X_std  = ckpt['X_std']
        trainer.train_losses = ckpt['train_losses']
        return trainer


def train_all_models(
    models: Dict[str, nn.Module],
    X: np.ndarray,
    y: np.ndarray,
    lengths: Optional[np.ndarray] = None,
    n_epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: Optional[str] = None,
    verbose: bool = True,
    use_freeze_schedule: bool = True,
    tau_lr_multiplier: float = 1.0,
    aux_variance_gamma: float = 0.0,
) -> Dict[str, Trainer]:
    """
    Train all models on the same data, return dict of Trainer objects.

    Args:
        models: {name: nn.Module} dict
        X, y: training data
        lengths: actual sequence lengths for loss masking
        n_epochs, lr, batch_size, device, verbose: training params
        use_freeze_schedule: if True, implement alternating freeze schedule for learnable timescales
        aux_variance_gamma: weight for auxiliary temporal variance penalty

    Returns:
        {name: Trainer} dict (each contains .train_losses and can .predict())
    """
    trainers = {}
    for name, model in models.items():
        if verbose:
            print(f"\n  Training {name}...")
        trainer = Trainer(model, lr=lr, batch_size=batch_size, device=device,
                          tau_lr_multiplier=tau_lr_multiplier)
        trainer.fit(X, y, n_epochs=n_epochs, lengths=lengths, verbose=verbose,
                    use_freeze_schedule=use_freeze_schedule,
                    aux_variance_gamma=aux_variance_gamma)
        trainers[name] = trainer
    return trainers
