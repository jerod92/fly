"""
anthill.train
-------------
Training loop infrastructure.

Primary API::

    from anthill.train import Trainer, TrainConfig
    from anthill.train.sanity import pre_training_check

    config = TrainConfig(epochs=30, lr=1e-3, batch_size=64)
    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.fit()
"""

from anthill.train.config import TrainConfig
from anthill.train.trainer import Trainer
from anthill.train import sanity

__all__ = ["Trainer", "TrainConfig", "sanity"]
