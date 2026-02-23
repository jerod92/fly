"""
fly.deploy
----------
Model export and inference wrapper utilities.

Usage::

    from fly.deploy import export, ModelWrapper
    export(model, "./my_model", format="torchscript", norm_stats={"mean": [...], "std": [...]})

    wrapper = ModelWrapper.load("./my_model")
    output = wrapper.predict(raw_input)
"""

from fly.deploy.export import export, ExportFormat
from fly.deploy.wrapper import ModelWrapper

__all__ = ["export", "ExportFormat", "ModelWrapper"]
