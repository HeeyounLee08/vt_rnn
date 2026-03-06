"""
train.py - Training loop for RNN models on Izhikevich spike data.

Provides a unified Trainer class that handles:
- Z-score normalization of inputs
- Poisson NLL loss (appropriate for spike count targets)
- Gradient clipping and LR scheduling
- Per-epoch loss tracking
"""

import numpy as np
from typing import Dict, List, Optional

import torch
import torch.nn as nn
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

    def _normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self.X_mean = X.mean(axis=(0, 1), keepdims=True)
            self.X_std = X.std(axis=(0, 1), keepdims=True)
        return (X - self.X_mean) / (self.X_std + 1e-8)

    def fit(self, X: np.ndarray, y: np.ndarray, n_epochs: int = 50,
            lengths: Optional[np.ndarray] = None,
            verbose: bool = True, use_freeze_schedule: bool = True) -> List[float]:
        """
        Train the model.

        Args:
            X: (n_trials, seq_len, n_features)
            y: (n_trials, seq_len) spike counts
            n_epochs: training epochs
            lengths: (n_trials,) actual sequence lengths for loss masking (None = no masking)
            verbose: print progress
            use_freeze_schedule: if True, implement alternating freeze schedule for learnable timescales

        Returns:
            List of per-epoch losses
        """
        # Detect if model has learnable timescales
        has_learnable_tau = any('rho' in name for name, _ in self.model.named_parameters())
        if not has_learnable_tau:
            use_freeze_schedule = False  # Don't freeze if no rho parameters exist

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
            (rho_params if 'rho' in pname else base_params).append(param)
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
                    # Phase 1: Freeze rho (timescales), train spatial weights
                    for name, param in self.model.named_parameters():
                        param.requires_grad = 'rho' not in name
                elif epoch < phase2_end:
                    # Phase 2: Train everything
                    for name, param in self.model.named_parameters():
                        param.requires_grad = True
                else:
                    # Phase 3: Freeze spatial weights, train rho (timescales)
                    for name, param in self.model.named_parameters():
                        param.requires_grad = 'rho' in name

                # Log phase transitions
                if verbose:
                    if epoch == phase1_end:
                        print(f"    Epoch {epoch+1}: Unfreezing timescales (Phase 2)")
                    elif epoch == phase2_end:
                        print(f"    Epoch {epoch+1}: Freezing spatial weights (Phase 3)")

            epoch_loss = 0.0
            for batch in loader:
                Xb, yb = batch[0], batch[1]
                optimizer.zero_grad()
                y_pred = self.model(Xb, return_sequence=True)  # (batch, seq_len)
                loss_per_step = y_pred - yb * torch.log(y_pred + 1e-8)
                if lengths is not None:
                    mask_b = batch[2]
                    loss = (loss_per_step * mask_b).sum() / mask_b.sum()
                else:
                    loss = loss_per_step.mean()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg = epoch_loss / len(loader)
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

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict firing rates for input sequences. Returns (n, seq_len)."""
        X_norm = self._normalize(X, fit=False)
        self.model.eval()
        X_t = torch.FloatTensor(X_norm).to(self.device)
        return self.model(X_t, return_sequence=True).cpu().numpy()


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
) -> Dict[str, Trainer]:
    """
    Train all models on the same data, return dict of Trainer objects.

    Args:
        models: {name: nn.Module} dict
        X, y: training data
        lengths: actual sequence lengths for loss masking
        n_epochs, lr, batch_size, device, verbose: training params
        use_freeze_schedule: if True, implement alternating freeze schedule for learnable timescales

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
                    use_freeze_schedule=use_freeze_schedule)
        trainers[name] = trainer
    return trainers
