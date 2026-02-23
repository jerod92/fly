"""
fly.train
---------
Training loop infrastructure.

Primary API::

    from fly.train import Trainer, TrainConfig
    from fly.train.sanity import pre_training_check

    config = TrainConfig(epochs=30, lr=1e-3, batch_size=64)
    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.fit()
"""

from fly.train.config import TrainConfig
from fly.train.trainer import Trainer
from fly.train import sanity

__all__ = ["Trainer", "TrainConfig", "sanity"]
