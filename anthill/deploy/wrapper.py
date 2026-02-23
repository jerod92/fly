"""
ModelWrapper: a minimal, stateless inference wrapper.

The agent should subclass this and implement preprocess() and postprocess()
for the specific task.  The base class handles loading and device management.
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch
import torch.nn as nn


class ModelWrapper:
    """
    Base class for a deployable model.

    Subclass and implement ``preprocess`` and ``postprocess``.
    The ``predict`` method applies the full pipeline:
      raw_input → preprocess → model forward → postprocess → raw_output

    This class is stateless and thread-safe (assuming the model is not mutated).

    Example::

        class ClockPredictor(ModelWrapper):
            def preprocess(self, raw_input):
                # raw_input is a PIL Image
                img = transforms.ToTensor()(raw_input)
                mean = torch.tensor(self.norm_stats["mean"])
                std  = torch.tensor(self.norm_stats["std"])
                return ((img - mean[:, None, None]) / std[:, None, None]).unsqueeze(0)

            def postprocess(self, model_output):
                hour   = model_output[0, 0].item() * 12
                minute = model_output[0, 1].item() * 60
                return {"hour": round(hour), "minute": round(minute)}

        predictor = ClockPredictor.load("./my_model", model_cls=ClockModel)
        result = predictor.predict(some_pil_image)
    """

    def __init__(self, model: nn.Module, norm_stats: dict, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.norm_stats = norm_stats
        self.device = device

    # ------------------------------------------------------------------
    # Override these in subclasses
    # ------------------------------------------------------------------

    def preprocess(self, raw_input: Any) -> torch.Tensor:
        """
        Convert raw input (PIL Image, string, numpy array, …) to a model-ready Tensor.
        Must include any normalization.  Output should have a batch dimension (shape [1, ...]).
        """
        raise NotImplementedError("Implement preprocess() in your ModelWrapper subclass.")

    def postprocess(self, model_output: torch.Tensor) -> Any:
        """
        Convert model output Tensor to the format the caller expects.
        Default: return the raw tensor.
        """
        return model_output

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def predict(self, raw_input: Any) -> Any:
        """Full pipeline: preprocess → forward → postprocess."""
        x = self.preprocess(raw_input)
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"preprocess() must return a Tensor, got {type(x)}")
        x = x.to(self.device)
        with torch.no_grad():
            out = self.model(x)
        return self.postprocess(out)

    # ------------------------------------------------------------------
    # Load / save
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        model_dir: str,
        model_cls: type[nn.Module] | None = None,
        model_kwargs: dict | None = None,
        device: str = "cpu",
    ) -> "ModelWrapper":
        """
        Load a model exported with ``anthill.deploy.export``.

        If ``model_cls`` is provided, loads state_dict into a new instance.
        Otherwise, attempts to load a TorchScript model.

        Parameters
        ----------
        model_dir : str
            Directory produced by anthill.deploy.export.
        model_cls : type | None
            The nn.Module class (required for state_dict format).
        model_kwargs : dict | None
            Constructor kwargs for model_cls.
        device : str
            Device to load the model onto.
        """
        # Load norm stats
        stats_path = os.path.join(model_dir, "normalization.json")
        norm_stats = {}
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                norm_stats = json.load(f)

        # Load model
        state_dict_path = os.path.join(model_dir, "model.pt")
        scripted_path = os.path.join(model_dir, "model_scripted.pt")

        if model_cls is not None and os.path.exists(state_dict_path):
            model = model_cls(**(model_kwargs or {}))
            model.load_state_dict(torch.load(state_dict_path, map_location=device))
        elif os.path.exists(scripted_path):
            model = torch.jit.load(scripted_path, map_location=device)
        else:
            raise FileNotFoundError(
                f"No model file found in {model_dir}.  "
                f"Expected model.pt or model_scripted.pt."
            )

        return cls(model=model, norm_stats=norm_stats, device=device)
