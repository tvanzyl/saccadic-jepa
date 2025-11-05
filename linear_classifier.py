from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import CrossEntropyLoss, Linear, Module, Parameter, Sequential
from torch.optim import SGD, Optimizer, Adam
from torch.optim.lr_scheduler import ExponentialLR

from lightly.utils.benchmarking.topk import mean_topk_accuracy
from lightly.utils.scheduler import CosineWarmupScheduler


class BaseClassifier(LightningModule, ABC):
    def __init__(
        self,
        model: Module,
        batch_size_per_device: int,
        lr: float,
        feature_dim: int,
        num_classes: int,
        topk: Tuple[int, ...],
    ) -> None:
       
        super().__init__()
        self.save_hyperparameters(ignore="model")

        self.model = model
        self.batch_size_per_device = batch_size_per_device
        self.lr = lr
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.topk = topk

        self.classification_head: Union[Linear, Sequential] = Linear(
            feature_dim, num_classes
        )
        self.criterion = CrossEntropyLoss()

    @abstractmethod
    def forward(self, images: Tensor) -> Tensor:
        """Implement in subclass."""
        pass

    def shared_step(
        self, batch: Tuple[Tensor, ...], batch_idx: int
    ) -> Tuple[Tensor, Dict[int, Tensor]]:
        images, targets = batch[0], batch[1]
        predictions = self.forward(images)
        loss = self.criterion(predictions, targets)
        _, predicted_labels = predictions.topk(max(self.topk))
        topk = mean_topk_accuracy(predicted_labels, targets, k=self.topk)

        return loss, topk

    def training_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> Tensor:
        loss, topk = self.shared_step(batch=batch, batch_idx=batch_idx)
        batch_size = len(batch[1])
        log_dict = {f"train_top{k}": acc for k, acc in topk.items()}
        self.log(
            "train_loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size
        )
        self.log_dict(log_dict, sync_dist=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> Tensor:
        loss, topk = self.shared_step(batch=batch, batch_idx=batch_idx)
        batch_size = len(batch[1])
        log_dict = {f"val_top{k}": acc for k, acc in topk.items()}
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log_dict(log_dict, prog_bar=True, sync_dist=True, batch_size=batch_size)

        return loss

    def get_effective_lr(self) -> float:
        """Compute the effective learning rate based on batch size and world size."""
        return self.lr * self.batch_size_per_device * self.trainer.world_size / 256

    @abstractmethod
    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Return the parameters that should be updated during training."""
        pass

    # Type ignore is needed because return type of LightningModule.configure_optimizers
    # is complicated and typing changes between versions.
    def configure_optimizers(  # type: ignore[override]
        self,
    ) -> Tuple[List[Optimizer], List[Dict[str, Union[Any, str]]]]:
        parameters = list(self.get_trainable_parameters())

        optimizer = SGD(
            parameters,
            lr=self.get_effective_lr(),
            momentum=0.9,
            weight_decay=0.0,
        )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=0,
                max_epochs=int(self.trainer.estimated_stepping_batches),
            ),
            "interval": "step",
        }

        return [optimizer], [scheduler]


class LinearClassifier(BaseClassifier):
    def __init__(
        self,
        model: Module,
        batch_size_per_device: int,
        lr: float = 0.1,
        feature_dim: int = 2048,
        num_classes: int = 1000,
        topk: Tuple[int, ...] = (1, 5),
    ) -> None:
        super().__init__(
            model=model,
            batch_size_per_device=batch_size_per_device,
            lr=lr,
            feature_dim=feature_dim,
            num_classes=num_classes,
            topk=topk,
        )

    def forward(self, images: Tensor) -> Tensor:
        # For linear evaluation, we want to freeze the feature extractor.
        with torch.no_grad():
            features = self.model(images).flatten(start_dim=1)

        output: Tensor = self.classification_head(features)

        return output

    def get_trainable_parameters(self) -> List[Parameter]:
        # Only update the classification head.
        return list(self.classification_head.parameters())

    def on_train_epoch_start(self) -> None:
        # Set model to eval mode to disable norm layer updates.
        self.model.eval()


class FinetuneClassifier(BaseClassifier):
    def __init__(
        self,
        model: Module,
        batch_size_per_device: int,
        lr: float = 0.05,
        feature_dim: int = 2048,
        num_classes: int = 1000,
        topk: Tuple[int, ...] = (1, 5),
    ) -> None:
        super().__init__(
            model=model,
            batch_size_per_device=batch_size_per_device,
            lr=lr,
            feature_dim=feature_dim,
            num_classes=num_classes,
            topk=topk,
        )

    def forward(self, images: Tensor) -> Tensor:
        # For finetuning, we want to update the feature extractor.
        features = self.model(images).flatten(start_dim=1)

        output: Tensor = self.classification_head(features)

        return output

    def get_trainable_parameters(self) -> List[Parameter]:
        # Update both the classification head and the feature extractor.
        return list(self.classification_head.parameters()) + list(
            self.model.parameters()
        )
