"""
examples/analogue_clock.py
--------------------------
End-to-end example: train a small CNN to read the hour from an analogue clock image.

This demonstrates the full anthill workflow:
  1. Procedural data generation (synthetic clock images, no external data needed)
  2. Small CNN trained from scratch
  3. anthill.train.Trainer with pre-training sanity check
  4. Model export and inference wrapper

Run with:
    python examples/analogue_clock.py

Trains in under 3 minutes on CPU.

Dependencies beyond anthill's base requirements:
    pip install Pillow
"""

from __future__ import annotations

import math
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import anthill
from anthill.train import Trainer, TrainConfig
from anthill.train.sanity import pre_training_check
from anthill.deploy import export, ModelWrapper


# ---------------------------------------------------------------------------
# Step 1: Workflow
# ---------------------------------------------------------------------------

wf = anthill.Workflow(task="Analogue clock → hour classification (0-11)")
wf.print_checklist()


# ---------------------------------------------------------------------------
# Step 2: Procedural data generator
# ---------------------------------------------------------------------------

class ClockDataset(Dataset):
    """
    Generates synthetic analogue clock images.

    Input:  (1, 32, 32) greyscale tensor, values in [0, 1]
    Output: integer label 0-11 (hour shown by the hour hand)

    Training split: varied hand lengths, angles, small noise
    Val split (OOD): clocks where the minute hand crosses the hour hand region,
                     making the task harder
    """

    IMG_SIZE = 32

    def __init__(self, split: str = "train", n_samples: int = 2000, seed: int = 42):
        assert split in ("train", "val", "test")
        self.split = split
        self.n_samples = n_samples
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        self.rng.seed(idx * 1000 + (0 if self.split == "train" else 1))
        hour = self.rng.randint(0, 11)
        img = self._render(hour)
        x = torch.from_numpy(img).float().unsqueeze(0)  # (1, H, W)
        y = torch.tensor(hour, dtype=torch.long)
        return x, y

    def _render(self, hour: int) -> "np.ndarray":
        import numpy as np

        size = self.IMG_SIZE
        img = np.zeros((size, size), dtype=np.float32)
        cx, cy = size // 2, size // 2
        r = size // 2 - 2

        # Clock face
        for y in range(size):
            for x in range(size):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                    img[y, x] = 0.85 + self.rng.uniform(-0.05, 0.05)

        # Hour hand
        hour_angle = math.radians(hour * 30 - 90)
        hour_len = r * (0.45 + self.rng.uniform(-0.05, 0.05))
        _draw_hand(img, cx, cy, hour_angle, hour_len, width=2)

        # Minute hand (random — adds noise, OOD val makes it cross hour hand)
        if self.split == "train":
            minute_angle = math.radians(self.rng.uniform(0, 360) - 90)
        else:
            # OOD: minute hand close to hour hand region
            offset = self.rng.uniform(-15, 15)
            minute_angle = math.radians(hour * 30 + offset - 90)
        minute_len = r * (0.65 + self.rng.uniform(-0.05, 0.05))
        _draw_hand(img, cx, cy, minute_angle, minute_len, width=1)

        # Gaussian noise
        noise = self.rng.gauss(0, 0.04)
        img += np.random.default_rng(self.rng.randint(0, 2**31)).normal(0, 0.04, img.shape)
        img = np.clip(img, 0, 1)
        return img


def _draw_hand(img, cx, cy, angle, length, width=1):
    import numpy as np
    dx, dy = math.cos(angle), math.sin(angle)
    for t in range(int(length)):
        x = int(cx + dx * t)
        y = int(cy + dy * t)
        for wx in range(-width // 2, width // 2 + 1):
            for wy in range(-width // 2, width // 2 + 1):
                nx, ny = x + wx, y + wy
                if 0 <= nx < img.shape[1] and 0 <= ny < img.shape[0]:
                    img[ny, nx] = 0.1


# ---------------------------------------------------------------------------
# Step 3: Model (small CNN, < 50k params)
# ---------------------------------------------------------------------------

class ClockCNN(nn.Module):
    """
    Small CNN for 32×32 greyscale clock images → 12-class output.
    ~28k parameters.  Trains on CPU in < 3 minutes.
    """

    def __init__(self, n_classes: int = 12):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.SiLU(),   # 32×32×16
            nn.Conv2d(16, 16, 3, padding=1), nn.SiLU(),
            nn.MaxPool2d(2),                               # 16×16×16
            nn.Conv2d(16, 32, 3, padding=1), nn.SiLU(),   # 16×16×32
            nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(),
            nn.MaxPool2d(2),                               # 8×8×32
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64), nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


# ---------------------------------------------------------------------------
# Step 4: Loss and metrics
# ---------------------------------------------------------------------------

loss_fn = nn.CrossEntropyLoss()


def compute_metrics(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    pred_labels = preds.argmax(dim=1)
    accuracy = (pred_labels == targets).float().mean().item()
    # Off-by-one: acceptable if the predicted hour is adjacent (clock ambiguity)
    off_by_one = ((pred_labels - targets).abs() % 12 <= 1).float().mean().item()
    return {
        "accuracy_pct": round(accuracy * 100, 1),
        "within_1_hour_pct": round(off_by_one * 100, 1),
    }


def format_sample(x: torch.Tensor, y_pred: torch.Tensor, y_true: torch.Tensor) -> str:
    pred = y_pred.argmax().item()
    true = y_true.item()
    ok = "✓" if pred == true else "✗"
    return f"{ok}  predicted={pred:2d}h  actual={true:2d}h"


# ---------------------------------------------------------------------------
# Step 5: Train
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("anthill example: Analogue Clock Hour Reader")
    print("=" * 60)

    train_ds = ClockDataset("train", n_samples=2000, seed=42)
    val_ds   = ClockDataset("val",   n_samples=400,  seed=99)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=64)

    model = ClockCNN()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: ClockCNN  ({params:,} trainable parameters)")

    config = TrainConfig(
        epochs=25,
        lr=1e-3,
        batch_size=64,
        device="auto",
        checkpoint_dir="./checkpoints/clock",
        early_stop_patience=8,
        log_every_n_steps=30,
        num_sample_predictions=4,
    )
    print(f"Device: {config.device}\n")

    # Sanity check before training
    ok = pre_training_check(
        model, train_loader, val_loader, config,
        loss_fn=loss_fn,
        expected_loss_range=(2.0, 2.6),  # ~log(12) for random weights
    )
    if not ok:
        print("Sanity check failed.  Fix issues before training.")
        exit(1)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        loss_fn=loss_fn,
        compute_metrics=compute_metrics,
        format_sample=format_sample,
    )
    trainer.fit()
    trainer.load_best()

    # Final test set evaluation
    test_ds = ClockDataset("test", n_samples=400, seed=7)
    test_loader = DataLoader(test_ds, batch_size=64)

    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            all_preds.append(model(x))
            all_targets.append(y)

    test_metrics = compute_metrics(torch.cat(all_preds), torch.cat(all_targets))
    print(f"\nTest set metrics: {test_metrics}")

    # Export
    export(
        model,
        output_dir="./clock_model",
        format="state_dict",
        metadata={"task": "analogue clock hour reader", "n_classes": 12},
    )

    # Mark workflow complete
    for key in wf._index:
        wf[key].complete()
    wf.print_checklist()
