"""
anthill.deploy
--------------
Model export and inference wrapper utilities.

Usage::

    from anthill.deploy import export, ModelWrapper
    export(model, "./my_model", format="torchscript", norm_stats={"mean": [...], "std": [...]})

    wrapper = ModelWrapper.load("./my_model")
    output = wrapper.predict(raw_input)
"""

from anthill.deploy.export import export, ExportFormat
from anthill.deploy.wrapper import ModelWrapper

__all__ = ["export", "ExportFormat", "ModelWrapper"]
