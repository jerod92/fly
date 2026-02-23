"""
anthill.train.trainer
---------------------
The Trainer class: a structured supervised training loop with rich feedback.

Features
--------
- Configurable via TrainConfig
- Callback system for extensibility
- Per-step progress with loss, gradient norm, samples/sec, and ETA
- Per-epoch validation with metrics and sample predictions
- Checkpoint saving on val_loss improvement
- Early stopping
- NaN watchdog (halts immediately on NaN loss or gradient)
- Learning rate scheduling (cosine, step)

Usage::

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        loss_fn=nn.CrossEntropyLoss(),
        compute_metrics=compute_metrics,  # optional: (preds, targets) -> dict
        format_sample=format_sample,      # optional: (x, y_pred, y_true) -> str
    )
    trainer.fit()
"""

from __future__ import annotations

import os
import time
from typing import Callable, Optional

import torch
import torch.nn as nn
from rich.console import Console

from anthill.train.config import TrainConfig
from anthill.train.callbacks import (
    Callback,
    TrainerState,
    CheckpointCallback,
    EarlyStoppingCallback,
    NaNWatchdog,
    RichProgressCallback,
)

console = Console()


class Trainer:
    """
    Supervised training loop.

    Parameters
    ----------
    model : nn.Module
    train_loader : DataLoader
    val_loader : DataLoader | None
        If None, validation is skipped (not recommended).
    config : TrainConfig
    loss_fn : Callable | None
        If None, the model's forward() must return a (output, loss) tuple.
    compute_metrics : Callable | None
        Signature: (preds: Tensor, targets: Tensor) -> dict[str, float]
        Called at the end of each validation epoch.
    format_sample : Callable | None
        Signature: (x: Tensor, y_pred: Tensor, y_true: Tensor) -> str
        Used to display sample predictions during validation.
        If None, a generic tensor summary is printed.
    extra_callbacks : list[Callback]
        Additional callbacks to run alongside the built-in ones.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader=None,
        config: Optional[TrainConfig] = None,
        loss_fn: Optional[Callable] = None,
        compute_metrics: Optional[Callable] = None,
        format_sample: Optional[Callable] = None,
        extra_callbacks: Optional[list[Callback]] = None,
    ):
        self.config = config or TrainConfig()
        self.model = model.to(torch.device(self.config.device))
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.compute_metrics = compute_metrics
        self.format_sample = format_sample

        # Built-in callbacks (always active)
        self.callbacks: list[Callback] = [
            NaNWatchdog(),
            RichProgressCallback(self.config.log_every_n_steps),
        ]
        if self.val_loader is not None:
            if self.config.early_stop_patience > 0:
                self.callbacks.append(EarlyStoppingCallback(self.config.early_stop_patience))
            self.callbacks.append(CheckpointCallback(
                self.config.checkpoint_dir, self.config.save_best_only
            ))
        if extra_callbacks:
            self.callbacks.extend(extra_callbacks)

        # State
        self._state = TrainerState()
        self._best_val_loss = float("inf")

        torch.manual_seed(self.config.seed)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def fit(self, epochs: Optional[int] = None) -> TrainerState:
        """Run the training loop.  Returns the final TrainerState."""
        cfg = self.config
        epochs = epochs or cfg.epochs
        device = torch.device(cfg.device)

        # Optimizer & scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        total_steps = epochs * len(self.train_loader)
        scheduler = _build_scheduler(optimizer, cfg, total_steps)

        state = self._state
        state.total_steps = total_steps
        state.lr = cfg.lr

        console.print()
        console.rule(f"[bold]Training — {epochs} epochs on {cfg.device}[/bold]")
        console.print(f"  Optimizer:   AdamW  lr={cfg.lr}  weight_decay={cfg.weight_decay}")
        console.print(f"  Schedule:    {cfg.lr_schedule}  warmup={cfg.warmup_steps} steps")
        console.print(f"  Checkpoints: {cfg.checkpoint_dir}")
        console.print()

        self._call("on_train_begin", state)

        step = 0
        for epoch in range(epochs):
            state.epoch = epoch
            state.lr = optimizer.param_groups[0]["lr"]
            self._call("on_epoch_begin", state)

            self.model.train()
            epoch_loss_sum = 0.0
            epoch_samples = 0
            epoch_t0 = time.time()

            for batch in self.train_loader:
                x, y = _unpack_batch(batch, device)
                batch_size = x.shape[0]
                t_batch = time.time()

                optimizer.zero_grad()
                loss, out = self._forward(x, y)
                loss.backward()

                grad_norm = 0.0
                if cfg.grad_clip > 0.0:
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.model.parameters(), cfg.grad_clip
                    ).item()

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                epoch_loss_sum += loss.item() * batch_size
                epoch_samples += batch_size
                elapsed_batch = time.time() - t_batch

                # Update state
                state.step = step
                state.train_loss = loss.item()
                state.grad_norm = grad_norm
                state.lr = optimizer.param_groups[0]["lr"]
                state.samples_per_sec = batch_size / max(elapsed_batch, 1e-6)

                # ETA estimates
                steps_done = step + 1
                steps_left = total_steps - steps_done
                avg_step_time = (time.time() - epoch_t0) / max(steps_done - epoch * len(self.train_loader), 1)
                steps_in_epoch = len(self.train_loader)
                steps_done_in_epoch = step - epoch * steps_in_epoch + 1
                state.eta_epoch_sec = (steps_in_epoch - steps_done_in_epoch) * avg_step_time
                state.eta_run_sec = steps_left * avg_step_time

                self._call("on_step_end", state)
                if state.should_stop:
                    break
                step += 1

            if state.should_stop:
                break

            state.train_loss = epoch_loss_sum / max(epoch_samples, 1)

            # Validation
            if self.val_loader is not None and (epoch + 1) % cfg.val_every_n_epochs == 0:
                val_loss, metrics = self._validate(device)
                state.val_loss = val_loss
                state.metrics = metrics

                improved = val_loss < self._best_val_loss
                state.best_val_loss = self._best_val_loss
                if improved:
                    self._best_val_loss = val_loss
                    state.best_val_loss = val_loss
                    self._save_checkpoint(epoch, optimizer)

                self._call("on_val_end", state)
                if state.should_stop:
                    break

            self._call("on_epoch_end", state)

        self._call("on_train_end", state)
        console.print(f"\n[bold green]Training complete.[/bold green]  "
                      f"Best val loss: {self._best_val_loss:.4f}")
        return state

    def load_best(self) -> None:
        """Load the best checkpoint saved during training."""
        path = os.path.join(self.config.checkpoint_dir, "best.pt")
        if not os.path.exists(path):
            console.print(f"[yellow]No checkpoint found at {path}[/yellow]")
            return
        ckpt = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        console.print(f"[green]Loaded best checkpoint (epoch {ckpt['epoch']}, "
                      f"val_loss={ckpt['val_loss']:.4f})[/green]")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(x)
        if isinstance(out, tuple) and len(out) == 2 and self.loss_fn is None:
            # Model returns (output, loss) directly
            return out[1], out[0]
        if self.loss_fn is None:
            raise ValueError(
                "Trainer requires either a loss_fn or a model that returns (output, loss)."
            )
        loss = self.loss_fn(out, y)
        return loss, out

    def _validate(
        self, device: torch.device
    ) -> tuple[float, dict]:
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
        sample_xs: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in self.val_loader:
                x, y = _unpack_batch(batch, device)
                loss, out = self._forward(x, y)
                total_loss += loss.item() * x.shape[0]
                total_samples += x.shape[0]
                all_preds.append(out.cpu())
                all_targets.append(y.cpu())
                if len(sample_xs) < self.config.num_sample_predictions:
                    sample_xs.append(x.cpu())

        avg_loss = total_loss / max(total_samples, 1)
        metrics: dict = {}

        if self.compute_metrics and all_preds:
            preds_cat = torch.cat(all_preds, dim=0)
            tgts_cat = torch.cat(all_targets, dim=0)
            metrics = self.compute_metrics(preds_cat, tgts_cat)

        # Display sample predictions
        if all_preds:
            self._print_sample_predictions(sample_xs, all_preds, all_targets)

        return avg_loss, metrics

    def _print_sample_predictions(self, xs, preds_list, targets_list) -> None:
        n = self.config.num_sample_predictions
        console.print(f"\n  [dim]Sample predictions (up to {n}):[/dim]")
        shown = 0
        for x_batch, p_batch, t_batch in zip(xs, preds_list, targets_list):
            for i in range(min(x_batch.shape[0], n - shown)):
                xi, pi, ti = x_batch[i], p_batch[i], t_batch[i]
                if self.format_sample:
                    line = self.format_sample(xi, pi, ti)
                else:
                    pred_label = pi.argmax().item() if pi.ndim > 0 else pi.item()
                    true_label = ti.item() if ti.numel() == 1 else ti.tolist()
                    correct = "✓" if pred_label == true_label else "✗"
                    line = f"  {correct}  pred={pred_label}  true={true_label}"
                console.print(f"  {line}")
                shown += 1
                if shown >= n:
                    return

    def _save_checkpoint(self, epoch: int, optimizer) -> None:
        path = os.path.join(self.config.checkpoint_dir, "best.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": self._best_val_loss,
            },
            path,
        )
        console.print(f"  [green]Checkpoint saved → {path}[/green]")

    def _call(self, hook: str, state: TrainerState) -> None:
        for cb in self.callbacks:
            getattr(cb, hook)(state)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unpack_batch(batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        x, y = batch[0], batch[1]
    else:
        raise ValueError(f"Expected batch to be (x, y), got {type(batch)}")
    x = x.to(device) if isinstance(x, torch.Tensor) else x
    y = y.to(device) if isinstance(y, torch.Tensor) else y
    return x, y


def _build_scheduler(optimizer, cfg: TrainConfig, total_steps: int):
    if cfg.lr_schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - cfg.warmup_steps
        )
    elif cfg.lr_schedule == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.step_lr_step_size, gamma=cfg.step_lr_gamma
        )
    return None
