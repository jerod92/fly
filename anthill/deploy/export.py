"""
Model export utilities.
"""

from __future__ import annotations

import json
import os
from enum import Enum
from typing import Any, Optional

import torch
import torch.nn as nn
from rich.console import Console

console = Console()


class ExportFormat(str, Enum):
    STATE_DICT = "state_dict"       # torch.save(state_dict) — Python only
    TORCHSCRIPT = "torchscript"     # torch.jit.script / trace
    ONNX = "onnx"                   # ONNX export


def export(
    model: nn.Module,
    output_dir: str,
    format: str = "state_dict",
    example_input: Optional[torch.Tensor] = None,
    norm_stats: Optional[dict] = None,
    metadata: Optional[dict] = None,
) -> str:
    """
    Export a trained model to ``output_dir``.

    Parameters
    ----------
    model : nn.Module
        Trained model in eval mode.
    output_dir : str
        Directory to write the model files.
    format : str
        One of 'state_dict', 'torchscript', 'onnx'.
    example_input : Tensor | None
        Required for 'torchscript' (trace) and 'onnx'.  A single example input
        with no batch dimension, or a batch of size 1.
    norm_stats : dict | None
        Normalization statistics (mean, std, vocab, etc.) to save alongside the model.
        Loaded automatically by ModelWrapper.
    metadata : dict | None
        Any additional metadata to store in metadata.json.

    Returns
    -------
    str
        Path to the primary output file.
    """
    fmt = ExportFormat(format.lower())
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    # Save normalization stats
    if norm_stats:
        stats_path = os.path.join(output_dir, "normalization.json")
        with open(stats_path, "w") as f:
            json.dump(norm_stats, f, indent=2)
        console.print(f"  Normalization stats → {stats_path}")

    # Save metadata
    import anthill
    meta = {
        "anthill_version": anthill.__version__,
        "format": fmt.value,
        **(metadata or {}),
    }
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Export model
    if fmt == ExportFormat.STATE_DICT:
        path = os.path.join(output_dir, "model.pt")
        torch.save(model.state_dict(), path)
        console.print(f"  [green]State dict → {path}[/green]")
        return path

    elif fmt == ExportFormat.TORCHSCRIPT:
        if example_input is None:
            raise ValueError("example_input is required for TorchScript export.")
        path = os.path.join(output_dir, "model_scripted.pt")
        try:
            scripted = torch.jit.script(model)
            console.print("  TorchScript: using torch.jit.script()")
        except Exception:
            if example_input is None:
                raise
            console.print("  TorchScript: jit.script failed, falling back to jit.trace()")
            scripted = torch.jit.trace(model, example_input)
        scripted.save(path)
        # Verify
        loaded = torch.jit.load(path)
        with torch.no_grad():
            ref = model(example_input)
            check = loaded(example_input)
        if not torch.allclose(ref, check, atol=1e-5):
            console.print("[yellow]Warning: TorchScript output differs slightly from original (>1e-5).[/yellow]")
        else:
            console.print(f"  [green]TorchScript → {path} (verified)[/green]")
        return path

    elif fmt == ExportFormat.ONNX:
        if example_input is None:
            raise ValueError("example_input is required for ONNX export.")
        path = os.path.join(output_dir, "model.onnx")
        torch.onnx.export(
            model,
            example_input,
            path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=17,
        )
        # Verify with onnxruntime if available
        try:
            import onnxruntime as ort
            import numpy as np
            sess = ort.InferenceSession(path)
            ort_out = sess.run(None, {"input": example_input.numpy()})[0]
            ref = model(example_input).detach().numpy()
            if not np.allclose(ref, ort_out, atol=1e-4):
                console.print("[yellow]Warning: ONNX output differs from PyTorch output (>1e-4).[/yellow]")
            else:
                console.print(f"  [green]ONNX → {path} (verified with onnxruntime)[/green]")
        except ImportError:
            console.print(f"  [green]ONNX → {path}[/green]  [dim](install onnxruntime to verify)[/dim]")
        return path

    raise ValueError(f"Unknown format: {fmt}")
