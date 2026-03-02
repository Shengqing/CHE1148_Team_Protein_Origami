from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .eval import regression_metrics, Metrics


@dataclass
class TrainResult:
    best_state_dict: dict
    best_val_mae: float


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    clip_grad_norm: float | None,
) -> Tuple[float, Metrics]:
    train_mode = optimizer is not None
    model.train(train_mode)

    losses = []
    y_true_list = []
    y_pred_list = []

    for x, y in tqdm(loader, leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x)
        loss = criterion(pred, y)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            optimizer.step()

        losses.append(loss.item())
        y_true_list.append(y.detach().cpu().numpy())
        y_pred_list.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    metrics = regression_metrics(y_true, y_pred)
    return float(np.mean(losses)), metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    clip_grad_norm: float,
    patience: int,
    factor: float,
) -> TrainResult:
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=factor, patience=patience
    )

    best_val_mae = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        tr_loss, tr_metrics = run_epoch(
            model, train_loader, device, criterion, optimizer, clip_grad_norm
        )
        va_loss, va_metrics = run_epoch(
            model, val_loader, device, criterion, optimizer=None, clip_grad_norm=None
        )

        scheduler.step(va_loss)

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {tr_loss:.4f} MAE {tr_metrics.mae:.4f} RMSE {tr_metrics.rmse:.4f} "
            f"R2 {tr_metrics.r2:.4f} Spearman {tr_metrics.spearman:.4f} | "
            f"val loss {va_loss:.4f} MAE {va_metrics.mae:.4f} RMSE {va_metrics.rmse:.4f} "
            f"R2 {va_metrics.r2:.4f} Spearman {va_metrics.spearman:.4f}"
        )

        if va_metrics.mae < best_val_mae:
            best_val_mae = va_metrics.mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training finished but no best_state was recorded.")

    return TrainResult(best_state_dict=best_state, best_val_mae=best_val_mae)