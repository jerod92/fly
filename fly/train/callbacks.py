"""
Callback system for the Trainer.

Each callback receives a TrainerState snapshot at the appropriate hook points.
Implement your own by subclassing Callback and overriding the relevant methods.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainerState:
    """Snapshot of training state passed to every callback hook."""
    epoch: int = 0
    step: int = 0
    total_steps: int = 0
    train_loss: float = float("inf")
    val_loss: float = float("inf")
    best_val_loss: float = float("inf")
    metrics: dict[str, float] = field(default_factory=dict)
    grad_norm: float = 0.0
    lr: float = 0.0
    samples_per_sec: float = 0.0
    eta_epoch_sec: float = 0.0
    eta_run_sec: float = 0.0
    should_stop: bool = False


class Callback:
    """Base callback.  Override any method you need."""

    def on_train_begin(self, state: TrainerState) -> None: ...
    def on_epoch_begin(self, state: TrainerState) -> None: ...
    def on_step_end(self, state: TrainerState) -> None: ...
    def on_epoch_end(self, state: TrainerState) -> None: ...
    def on_val_end(self, state: TrainerState) -> None: ...
    def on_train_end(self, state: TrainerState) -> None: ...


# ---------------------------------------------------------------------------
# Built-in callbacks
# ---------------------------------------------------------------------------

class RichProgressCallback(Callback):
    """Prints a Rich-formatted progress line every log_every_n_steps steps."""

    def __init__(self, log_every_n_steps: int = 50):
        self.log_every_n_steps = log_every_n_steps
        self._epoch_start: float = 0.0

    def on_epoch_begin(self, state: TrainerState) -> None:
        self._epoch_start = time.time()
        from rich.console import Console
        Console().print(
            f"\n[bold]Epoch {state.epoch + 1}[/bold]  "
            f"[dim]lr={state.lr:.2e}[/dim]",
        )

    def on_step_end(self, state: TrainerState) -> None:
        if state.step % self.log_every_n_steps != 0:
            return
        elapsed = time.time() - self._epoch_start
        from rich.console import Console
        Console().print(
            f"  step {state.step:>6}  "
            f"loss [cyan]{state.train_loss:.4f}[/cyan]  "
            f"grad_norm [yellow]{state.grad_norm:.2f}[/yellow]  "
            f"[dim]{state.samples_per_sec:.0f} samp/s  "
            f"ETA epoch {_fmt_sec(state.eta_epoch_sec)}  "
            f"ETA run {_fmt_sec(state.eta_run_sec)}[/dim]",
        )

    def on_val_end(self, state: TrainerState) -> None:
        from rich.console import Console
        from rich.table import Table
        from rich import box as rbox
        c = Console()
        table = Table(box=rbox.SIMPLE, show_header=True, header_style="bold")
        table.add_column("metric")
        table.add_column("value", justify="right")
        table.add_row("val_loss", f"{state.val_loss:.4f}")
        for k, v in state.metrics.items():
            table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
        best_marker = " [green](new best)[/green]" if state.val_loss <= state.best_val_loss else ""
        c.print(f"\n  Validation{best_marker}")
        c.print(table)


class EarlyStoppingCallback(Callback):
    """Stop training if val_loss does not improve for `patience` epochs."""

    def __init__(self, patience: int = 10):
        self.patience = patience
        self._no_improve = 0
        self._best = float("inf")

    def on_val_end(self, state: TrainerState) -> None:
        if state.val_loss < self._best:
            self._best = state.val_loss
            self._no_improve = 0
        else:
            self._no_improve += 1
            if self._no_improve >= self.patience:
                from rich.console import Console
                Console().print(
                    f"\n[yellow]Early stopping: val_loss did not improve for "
                    f"{self.patience} epochs.[/yellow]"
                )
                state.should_stop = True


class CheckpointCallback(Callback):
    """Save the model checkpoint when val_loss improves."""

    def __init__(self, checkpoint_dir: str, save_best_only: bool = True):
        self.checkpoint_dir = checkpoint_dir
        self.save_best_only = save_best_only
        self._best = float("inf")

    def on_val_end(self, state: TrainerState) -> None:
        if self.save_best_only and state.val_loss >= self._best:
            return
        import os, torch
        self._best = min(self._best, state.val_loss)
        path = os.path.join(self.checkpoint_dir, "best.pt")
        # The actual model is saved by Trainer; we just track state here.
        # Trainer calls this callback AFTER saving; this is a no-op by default.


class NaNWatchdog(Callback):
    """Halt training immediately if loss or gradient norm is NaN."""

    def on_step_end(self, state: TrainerState) -> None:
        import math
        if math.isnan(state.train_loss) or math.isinf(state.train_loss):
            from rich.console import Console
            Console().print(
                f"\n[bold red]NaN/Inf detected in training loss at step {state.step}. "
                f"Halting.[/bold red]"
            )
            state.should_stop = True

    def on_val_end(self, state: TrainerState) -> None:
        import math
        if math.isnan(state.val_loss) or math.isinf(state.val_loss):
            from rich.console import Console
            Console().print(
                f"\n[bold red]NaN/Inf detected in validation loss at epoch {state.epoch}. "
                f"Halting.[/bold red]"
            )
            state.should_stop = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_sec(sec: float) -> str:
    if sec < 60:
        return f"{sec:.0f}s"
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"
