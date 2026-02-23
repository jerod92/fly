"""Training configuration dataclass."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    """
    Hyperparameters and settings for the training loop.

    All fields have sensible defaults that work for a first run.
    Adjust based on task complexity and hardware.
    """

    # --- Core hyperparameters ---
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # --- Learning rate schedule ---
    lr_schedule: str = "cosine"         # "cosine" | "step" | "none"
    warmup_steps: int = 0               # 0 = no warmup; set to ~5% of total steps
    step_lr_step_size: int = 10         # only used when lr_schedule="step"
    step_lr_gamma: float = 0.1          # only used when lr_schedule="step"

    # --- Regularization ---
    grad_clip: float = 1.0              # 0.0 = disabled

    # --- Hardware ---
    device: str = "auto"                # "auto" | "cpu" | "cuda" | "mps"

    # --- Logging ---
    log_every_n_steps: int = 50         # print/log every N optimizer steps
    val_every_n_epochs: int = 1         # run validation every N epochs
    num_sample_predictions: int = 4     # print this many sample preds at each val

    # --- Checkpointing ---
    checkpoint_dir: str = "./checkpoints"
    save_best_only: bool = True         # only save when val_loss improves

    # --- Early stopping ---
    early_stop_patience: int = 10       # 0 = disabled; stop if val_loss doesn't improve

    # --- Misc ---
    seed: int = 42

    def __post_init__(self) -> None:
        if self.lr_schedule not in ("cosine", "step", "none"):
            raise ValueError(f"lr_schedule must be 'cosine', 'step', or 'none', got {self.lr_schedule!r}")
        if self.device == "auto":
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def summary(self) -> str:
        lines = ["TrainConfig:"]
        for k, v in self.__dict__.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)
