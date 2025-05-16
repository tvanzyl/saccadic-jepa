from pathlib import Path
from typing import Dict, Sequence, Union

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms as T

from lightly.data import LightlyDataset
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.utils.benchmarking import KNNClassifier, MetricCallback
from lightly.utils.dist import print_rank_zero


def knn_eval(
    model: LightningModule,
    train_dir: Path,
    val_dir: Path,
    log_dir: Path,
    batch_size_per_device: int,
    num_workers: int,
    accelerator: str,
    devices: int,
    num_classes: int,
    knn_k: Union[Sequence[int], int],
    transform
) -> Dict[str, float]:
    """Runs KNN evaluation on the given model.

    Parameters follow InstDisc [0] settings.

    The most important settings are:
        - Num nearest neighbors: 200
        - Temperature: 0.1

    References:
       - [0]: InstDict, 2018, https://arxiv.org/abs/1805.01978
    """
    print_rank_zero("Running KNN evaluation...")

    train_dataset = LightlyDataset(input_dir=str(train_dir), transform=transform)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_device,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    # Setup validation data.
    val_dataset = LightlyDataset(input_dir=str(val_dir), transform=transform)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_per_device,
        shuffle=False,
        num_workers=num_workers,
    )

    metrics_dict: dict[str, float] = dict()
    for k in knn_k:
        classifier = KNNClassifier(
            model=model,
            num_classes=num_classes,
            feature_dtype=torch.float16,
            knn_k=k,
        )

        # Run KNN evaluation.
        metric_callback = MetricCallback()
        trainer = Trainer(
            enable_checkpointing=False,
            max_epochs=1,
            accelerator=accelerator,
            devices=devices,
            logger=False, #TensorBoardLogger(save_dir=str(log_dir), name="knn_eval", version=k),
            callbacks=[
                # DeviceStatsMonitor(),
                metric_callback,
            ],
            strategy="ddp_find_unused_parameters_true",
            num_sanity_val_steps=0,
        )
        trainer.fit(
            model=classifier,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )        
        for metric in ["val_top1", "val_top5"]:
            print(f"knn-{k} {metric}: {max(metric_callback.val_metrics[metric])}")
            metrics_dict[metric+f"@{k}"] = max(metric_callback.val_metrics[metric])
    return metrics_dict
