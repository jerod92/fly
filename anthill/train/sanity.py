"""
anthill.train.sanity
--------------------
Pre-training sanity checks.  Run these BEFORE any full training run.

Usage::

    from anthill.train.sanity import pre_training_check, check_dataset
    pre_training_check(model, train_loader, val_loader, config)
"""

from __future__ import annotations

import math
import time
from typing import Callable, Optional

import torch
import torch.nn as nn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def pre_training_check(
    model: nn.Module,
    train_loader,
    val_loader,
    config,
    loss_fn: Optional[Callable] = None,
    expected_loss_range: Optional[tuple[float, float]] = None,
) -> bool:
    """
    Run all pre-training checks.  Returns True if all pass, False if any fail.

    Checks performed:
      1. Single batch forward + backward (shape, dtype, NaN)
      2. Loss plausibility (optional, if expected_loss_range given)
      3. Overfit-one-batch test
      4. Data loading speed
      5. Sample display
    """
    console.print()
    console.rule("[bold cyan]Pre-training sanity check[/bold cyan]")

    results: list[tuple[str, bool, str]] = []

    device = torch.device(config.device)
    model = model.to(device)

    # Grab one batch
    try:
        batch = next(iter(train_loader))
    except StopIteration:
        console.print("[bold red]FAIL: train_loader is empty.[/bold red]")
        return False

    x, y = _unpack_batch(batch, device)

    # --- Check 1: forward + backward ---
    ok, msg = _check_forward_backward(model, x, y, loss_fn)
    results.append(("Forward + backward pass", ok, msg))

    # --- Check 2: loss plausibility ---
    if expected_loss_range and loss_fn:
        ok, msg = _check_loss_plausibility(model, x, y, loss_fn, expected_loss_range)
        results.append(("Loss plausibility", ok, msg))

    # --- Check 3: overfit one batch ---
    if loss_fn:
        ok, msg = _check_overfit_one_batch(model, x, y, loss_fn, config)
        results.append(("Overfit one batch", ok, msg))

    # --- Check 4: data loading speed ---
    ok, msg = _check_loader_speed(train_loader, device)
    results.append(("Data loading speed", ok, msg))

    # --- Check 5: dataset validation ---
    ok, msg = check_dataset(train_loader.dataset, quiet=False)
    results.append(("Dataset validation", ok, msg))

    # --- Print results table ---
    _print_results(results)

    all_passed = all(r[1] for r in results)
    if all_passed:
        console.print("[bold green]All checks passed.  Safe to begin training.[/bold green]\n")
    else:
        console.print(
            "[bold red]One or more checks failed.  "
            "Fix the issues above before running a full training loop.[/bold red]\n"
        )
    return all_passed


def check_dataset(dataset, n_display: int = 4, quiet: bool = True) -> tuple[bool, str]:
    """
    Validate a dataset object.

    Checks: length, sample shape consistency, NaN/Inf values, label distribution.
    Returns (passed: bool, message: str).
    """
    issues: list[str] = []
    warnings: list[str] = []

    if len(dataset) == 0:
        return False, "Dataset is empty."

    # Sample a few items
    indices = list(range(min(n_display, len(dataset))))
    samples = [dataset[i] for i in indices]
    xs = [s[0] if isinstance(s, (tuple, list)) else s for s in samples]
    ys = [s[1] if isinstance(s, (tuple, list)) else None for s in samples]

    # Shape consistency
    shapes = [x.shape if hasattr(x, "shape") else None for x in xs]
    if shapes[0] is not None and len(set(str(s) for s in shapes)) > 1:
        issues.append(f"Inconsistent sample shapes: {shapes}")

    # NaN / Inf
    for i, x in enumerate(xs):
        if isinstance(x, torch.Tensor):
            if torch.isnan(x).any():
                issues.append(f"NaN in sample {i}")
            if torch.isinf(x).any():
                issues.append(f"Inf in sample {i}")

    # Label distribution (basic)
    if ys[0] is not None:
        label_sample = [y.item() if isinstance(y, torch.Tensor) and y.numel() == 1 else str(y) for y in ys]
        if not quiet:
            console.print(f"  [dim]Sample labels (first {n_display}): {label_sample}[/dim]")

    if issues:
        return False, "; ".join(issues)
    msg = f"OK — {len(dataset)} samples, shape {shapes[0]}"
    if warnings:
        msg += "; warnings: " + "; ".join(warnings)
    return True, msg


def estimate_memory(
    model: nn.Module,
    input_shape: tuple[int, ...],
    batch_size: int = 32,
    dtype: torch.dtype = torch.float32,
) -> dict[str, str]:
    """
    Rough estimate of memory usage for model + one batch.

    Returns a dict with 'params_mb', 'batch_mb', 'total_mb'.
    """
    bytes_per_element = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
    }.get(dtype, 4)

    param_count = sum(p.numel() for p in model.parameters())
    # Parameters + gradients + optimizer state (Adam = 2x params)
    params_mb = param_count * bytes_per_element * 4 / 1e6

    batch_elements = batch_size * math.prod(input_shape)
    batch_mb = batch_elements * bytes_per_element / 1e6

    return {
        "params_mb": f"{params_mb:.1f} MB",
        "batch_mb": f"{batch_mb:.1f} MB",
        "total_mb": f"{params_mb + batch_mb:.1f} MB",
        "param_count": f"{param_count:,}",
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _unpack_batch(batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        x, y = batch[0], batch[1]
    else:
        raise ValueError(f"Expected batch to be a (x, y) tuple, got {type(batch)}")
    if isinstance(x, torch.Tensor):
        x = x.to(device)
    if isinstance(y, torch.Tensor):
        y = y.to(device)
    return x, y


def _check_forward_backward(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: Optional[Callable],
) -> tuple[bool, str]:
    try:
        model.train()
        out = model(x)
        if torch.isnan(out).any():
            return False, f"NaN in model output (shape {out.shape})"
        if torch.isinf(out).any():
            return False, f"Inf in model output (shape {out.shape})"
        if loss_fn:
            loss = loss_fn(out, y)
            loss.backward()
            model.zero_grad()
        return True, f"output shape {out.shape}, loss computable"
    except Exception as e:
        return False, str(e)


def _check_loss_plausibility(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: Callable,
    expected_range: tuple[float, float],
) -> tuple[bool, str]:
    try:
        model.eval()
        with torch.no_grad():
            out = model(x)
            loss = loss_fn(out, y).item()
        lo, hi = expected_range
        if lo <= loss <= hi:
            return True, f"loss={loss:.4f} (expected [{lo:.2f}, {hi:.2f}])"
        else:
            return False, (
                f"loss={loss:.4f} is outside expected range [{lo:.2f}, {hi:.2f}] "
                f"for random weights.  Check loss function and label format."
            )
    except Exception as e:
        return False, str(e)


def _check_overfit_one_batch(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: Callable,
    config,
    n_steps: int = 40,
    target_loss: float = 0.01,
) -> tuple[bool, str]:
    """Train on a single batch for n_steps.  Loss should drop to near-zero."""
    import copy
    m = copy.deepcopy(model)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    m.train()
    initial_loss = None
    final_loss = None
    for step in range(n_steps):
        opt.zero_grad()
        out = m(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    if final_loss is None or initial_loss is None:
        return False, "Could not compute loss"

    reduction = (initial_loss - final_loss) / (initial_loss + 1e-8)
    if final_loss <= target_loss:
        return True, f"loss {initial_loss:.4f} → {final_loss:.4f} in {n_steps} steps (overfit confirmed)"
    elif reduction > 0.5:
        return True, (
            f"loss {initial_loss:.4f} → {final_loss:.4f} in {n_steps} steps "
            f"({reduction*100:.0f}% reduction — didn't reach {target_loss} but trending correctly)"
        )
    else:
        return False, (
            f"loss barely changed: {initial_loss:.4f} → {final_loss:.4f} after {n_steps} steps. "
            f"The model may not be expressive enough, or there is a bug in the loss / labels."
        )


def _check_loader_speed(loader, device: torch.device, n_batches: int = 10) -> tuple[bool, str]:
    t0 = time.time()
    count = 0
    total_samples = 0
    for batch in loader:
        x, _ = _unpack_batch(batch, device)
        total_samples += x.shape[0]
        count += 1
        if count >= n_batches:
            break
    elapsed = time.time() - t0
    ms_per_batch = elapsed / count * 1000
    if ms_per_batch > 200:
        return False, (
            f"{ms_per_batch:.0f} ms/batch — data loading is a bottleneck. "
            f"Consider increasing num_workers or caching data."
        )
    return True, f"{ms_per_batch:.0f} ms/batch ({total_samples/elapsed:.0f} samples/sec)"


def _print_results(results: list[tuple[str, bool, str]]) -> None:
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("check")
    table.add_column("result", width=6)
    table.add_column("details")
    for name, passed, msg in results:
        icon = "[green]PASS[/green]" if passed else "[bold red]FAIL[/bold red]"
        table.add_row(name, icon, msg)
    console.print(table)
