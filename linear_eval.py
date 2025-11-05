from pathlib import Path
from typing import Dict

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import DeviceStatsMonitor, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import Module
from torch.utils.data import DataLoader
from torchvision import transforms as T

from lightly.data import LightlyDataset
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.utils.benchmarking import MetricCallback
# from lightly.utils.benchmarking import LinearClassifier
from linear_classifier import LinearClassifier
from lightly.utils.dist import print_rank_zero


def linear_eval(
    model: Module,
    train_dir: Path,
    val_dir: Path,
    log_dir: Path,
    batch_size_per_device: int,
    num_workers: int,
    accelerator: str,
    devices: int,
    precision: str,
    num_classes: int, 
    train_transform,
    val_transform,
    linear_lr,
) -> Dict[str, float]:
    """Runs a linear evaluation on the given model.

    Parameters follow SimCLR [0] settings.

    The most important settings are:
        - Backbone: Frozen
        - Epochs: 100
        - Optimizer: SGD
        - Base Learning Rate: 0.1
        - Momentum: 0.9
        - Weight Decay: 0.0
        - LR Schedule: Cosine without warmup

    References:
        - [0]: SimCLR, 2020, https://arxiv.org/abs/2002.05709
    """
    print_rank_zero("Running linear evaluation...")
    feature_dim = model.emb_width

    # Setup training data.
    train_dataset = LightlyDataset(input_dir=str(train_dir), transform=train_transform)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_device,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        persistent_workers=False,
    )

    # Setup validation data.    
    val_dataset = LightlyDataset(input_dir=str(val_dir), transform=val_transform)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_per_device,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=False,
    )
    # Train linear classifier.
    metric_callback = MetricCallback()
    trainer = Trainer(
        max_epochs=100,
        accelerator=accelerator,
        devices=devices,
        callbacks=[
            LearningRateMonitor(),
            DeviceStatsMonitor(),
            metric_callback,
        ],
        logger=TensorBoardLogger(save_dir=str(log_dir), name="linear_eval"),
        precision=precision,
        strategy="ddp_find_unused_parameters_true",
        num_sanity_val_steps=0,
    )
    classifier = LinearClassifier(
        model=model,
        batch_size_per_device=batch_size_per_device,
        feature_dim=feature_dim,
        num_classes=num_classes,
        # freeze_model=True,
        lr=linear_lr        
    )
    trainer.fit(
        model=classifier,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    metrics_dict: Dict[str, float] = dict()
    for metric in ["val_top1", "val_top5"]:
        print(f"max linear {metric}: {max(metric_callback.val_metrics[metric])}")
        metrics_dict[metric] = max(metric_callback.val_metrics[metric])
    return metrics_dict
