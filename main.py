from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Dict, Sequence, Union

import SimPLR
from SimPLR import train_transform, dataset_with_indices
import finetune_eval
import knn_eval
import linear_eval
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms as T

from lightly.data import LightlyDataset
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.utils.benchmarking import MetricCallback
from lightly.utils.dist import print_rank_zero

parser = ArgumentParser("ImageNet ResNet50 Benchmarks")
parser.add_argument("--train-dir", type=Path, default="/media/tvanzyl/data/imagenet/train")
parser.add_argument("--val-dir", type=Path, default="/media/tvanzyl/data/imagenet/val")
parser.add_argument("--log-dir", type=Path, default="benchmark_logs")
parser.add_argument("--batch-size-per-device", type=int, default=256)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--num-workers", type=int, default=8)
parser.add_argument("--accelerator", type=str, default="gpu")
parser.add_argument("--devices", type=int, default=1)
parser.add_argument("--backbone", type=str, default="resnet-50")
parser.add_argument("--precision", type=str, default="16-mixed")
parser.add_argument("--ckpt-path", type=Path, default=None)
parser.add_argument("--compile-model", action="store_true")
parser.add_argument("--methods", type=str, nargs="+")
parser.add_argument("--num-classes", type=int, default=1000)
parser.add_argument("--skip-knn-eval", action="store_true")
parser.add_argument("--skip-linear-eval", action="store_true")
parser.add_argument("--skip-finetune-eval", action="store_true")
parser.add_argument("--knn-k", type=int, nargs="+")
parser.add_argument("--lr", type=float, default=0.15)
parser.add_argument("--decay", type=float, default=1e-4)
parser.add_argument("--running-stats", action="store_true")
parser.add_argument("--ema-v2", action="store_true")
parser.add_argument("--momentum-head", action="store_true")
parser.add_argument("--identity-head", action="store_true")
parser.add_argument("--no-projection-head", action="store_true")
parser.add_argument("--m", type=float, default=0.5)
parser.add_argument("--linear-lr", type=float, default=0.1)

METHODS = {
    "Cifar10":      {"model": SimPLR.SimPLR, "n_local_views":0,
                     "train_transform": train_transform(32, (0.2, 1.)),
                     "val_transform": SimPLR.val_transforms["Cifar10"],  
                     "transform": SimPLR.transforms["Cifar10"],},
    "Cifar100":     {"model": SimPLR.SimPLR, "n_local_views":0,
                     "train_transform": train_transform(32, (0.2, 1.)),
                     "val_transform": SimPLR.val_transforms["Cifar100"], 
                     "transform": SimPLR.transforms["Cifar100"],},

    "Tiny":         {"model": SimPLR.SimPLR, "n_local_views":6,
                     "train_transform": train_transform(64, (0.2, 1.)),
                     "val_transform": SimPLR.val_transforms["Tiny"],
                     "transform": SimPLR.transforms["Tiny"],},
    "Tiny-64-W":   {"model": SimPLR.SimPLR, "n_local_views":0,
                     "train_transform": train_transform(64, (0.2, 1.)),
                     "val_transform": SimPLR.val_transforms["Tiny"],
                     "transform": SimPLR.transforms["Tiny-64-W"],},
    "Tiny-64-S":   {"model": SimPLR.SimPLR, "n_local_views":0,
                     "train_transform": train_transform(64, (0.2, 1.)),
                     "val_transform": SimPLR.val_transforms["Tiny"],
                     "transform": SimPLR.transforms["Tiny-64-S"],},

    "STL":          {"model": SimPLR.SimPLR, "n_local_views":6,
                     "train_transform": train_transform(96, (0.2, 1.)),
                     "val_transform": SimPLR.val_transforms["STL"],
                     "transform": SimPLR.transforms["STL"],},
    "STL-S":        {"model": SimPLR.SimPLR, "n_local_views":0,
                     "train_transform": train_transform(96, (0.2, 1.)),
                     "val_transform": SimPLR.val_transforms["STL"],
                     "transform": SimPLR.transforms["STL-S"],},

    "Nette":        {"model": SimPLR.SimPLR, "n_local_views":6,
                     "train_transform": train_transform(128),
                     "val_transform": SimPLR.val_transforms["Nette"],    
                     "transform": SimPLR.transforms["Nette"],},
    
    "Im100":        {"model": SimPLR.SimPLR, "n_local_views":6,
                     "train_transform": train_transform(224),
                     "val_transform": SimPLR.val_transforms["Im100"],    
                     "transform": SimPLR.transforms["Im100"],},
    "Im100-2":      {"model": SimPLR.SimPLR, "n_local_views":0,
                     "train_transform": train_transform(224),
                     "val_transform": SimPLR.val_transforms["Im100"],  
                     "transform": SimPLR.transforms["Im100-2"],},

    "Im1k":         {"model": SimPLR.SimPLR, "n_local_views":6,
                     "train_transform": train_transform(224),
                     "val_transform": SimPLR.val_transforms["Im1k"],     
                     "transform": SimPLR.transforms["Im1k"],},
    "Im1k-2":       {"model": SimPLR.SimPLR, "n_local_views":0,
                     "train_transform": train_transform(224),
                     "val_transform": SimPLR.val_transforms["Im1k"],   
                     "transform": SimPLR.transforms["Im1k-2"],},
}

def main(
    train_dir: Path,
    val_dir: Path,
    log_dir: Path,
    batch_size_per_device: int,
    epochs: int,
    num_workers: int,
    accelerator: str,
    devices: int,
    backbone: str,
    precision: str,
    compile_model: bool,
    methods: Union[Sequence[str], None],
    num_classes: int,
    skip_knn_eval: bool,
    skip_linear_eval: bool,
    skip_finetune_eval: bool,
    ckpt_path: Union[Path, None],
    knn_k: Union[Sequence[int], int],
    lr: float,
    decay: float,
    running_stats: bool,
    ema_v2: bool,
    momentum_head: bool,
    identity_head: bool,
    no_projection_head: bool,
    m:float,
    linear_lr:float,
) -> None:
    torch.set_float32_matmul_precision("high")

    method_names = methods or METHODS.keys()
    knn_k = knn_k or [1, 2, 5, 10, 20, 50, 100, 200]
    knn_k = knn_k if isinstance(knn_k,list) else [knn_k]

    for method in method_names:
        method_dir = (
            log_dir / method / datetime.now().strftime("%m-%d_%H-%M")
        ).resolve()
        if ckpt_path is not None: #Rename method dir if pretrain exists
            paths = ckpt_path.split("pretrain")
            if len(paths) > 0:
                method_dir = Path(paths[0]).resolve()
        
        model = METHODS[method]["model"](
            batch_size_per_device=batch_size_per_device, 
            num_classes=num_classes, 
            backbone=backbone,
            n_local_views=METHODS[method]["n_local_views"],
            lr=lr,
            decay=decay,
            running_stats=running_stats,
            ema_v2=ema_v2,
            momentum_head=momentum_head,
            identity_head=identity_head,
            no_projection_head=no_projection_head,
            m=m,
        )

        if compile_model and hasattr(torch, "compile"):
            # Compile model if PyTorch supports it.
            print_rank_zero("Compiling model...")
            model = torch.compile(model)

        if epochs <= 0:
            print_rank_zero("Epochs <= 0, skipping pretraining.")
            if ckpt_path is not None:
                model.load_state_dict(torch.load(ckpt_path)["state_dict"])
        else:
            pretrain(
                model=model,
                method=method,
                train_dir=train_dir,
                val_dir=val_dir,
                log_dir=method_dir,
                batch_size_per_device=batch_size_per_device,
                epochs=epochs,
                num_workers=num_workers,
                accelerator=accelerator,
                devices=devices,
                precision=precision,
                ckpt_path=ckpt_path,                
            )
        eval_metrics: Dict[str, Dict[str, float]] = dict()
        if skip_knn_eval:
            print_rank_zero("Skipping KNN eval.")
        else:
            eval_metrics["knn"] = knn_eval.knn_eval(
                model=model,
                num_classes=num_classes,
                train_dir=train_dir,
                val_dir=val_dir,
                log_dir=method_dir,
                batch_size_per_device=batch_size_per_device,
                num_workers=num_workers,
                accelerator=accelerator,
                devices=devices,
                knn_k=knn_k,
                transform=METHODS[method]["val_transform"]
            )

        if skip_linear_eval:
            print_rank_zero("Skipping linear eval.")
        else:            
            eval_metrics["linear"] = linear_eval.linear_eval(
                model=model,
                num_classes=num_classes,
                train_dir=train_dir,
                val_dir=val_dir,
                log_dir=method_dir,
                batch_size_per_device=batch_size_per_device,
                num_workers=num_workers,
                accelerator=accelerator,
                devices=devices,
                precision=precision,
                train_transform=METHODS[method]["train_transform"],
                val_transform=METHODS[method]["val_transform"],
                linear_lr=linear_lr,
            )

        if skip_finetune_eval:
            print_rank_zero("Skipping fine-tune eval.")
        else:
            eval_metrics["finetune"] = finetune_eval.finetune_eval(
                model=model,
                num_classes=num_classes,
                train_dir=train_dir,
                val_dir=val_dir,
                log_dir=method_dir,
                batch_size_per_device=batch_size_per_device,
                num_workers=num_workers,
                accelerator=accelerator,
                devices=devices,
                precision=precision,
                train_transform=METHODS[method]["train_transform"],
                val_transform=METHODS[method]["val_transform"]
            )

        if eval_metrics:
            print(f"Results for {method}:")
            print(eval_metrics_to_markdown(eval_metrics))


def pretrain(
    model: LightningModule,
    method: str,
    train_dir: Path,
    val_dir: Path,
    log_dir: Path,
    batch_size_per_device: int,
    epochs: int,
    num_workers: int,
    accelerator: str,
    devices: int,
    precision: str,
    ckpt_path: Union[Path, None],
) -> None:
    print_rank_zero(f"Running pretraining for {method}...")

    # Setup training data.
    train_transform = METHODS[method]["transform"]
    train_dataset = dataset_with_indices(LightlyDataset)(input_dir=str(train_dir), transform=train_transform)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_device,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        persistent_workers=False,
    )

    # Setup validation data.
    val_transform = METHODS[method]["val_transform"]

    val_dataset = LightlyDataset(input_dir=str(val_dir), transform=val_transform)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_per_device,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=False,
    )

    # Train model.
    metric_callback = MetricCallback()
    trainer = Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=[
            LearningRateMonitor(),
            # Stop if training loss diverges.
            EarlyStopping(monitor="train_loss", patience=int(1e12), check_finite=True),
            DeviceStatsMonitor(),
            metric_callback,
        ],
        logger=TensorBoardLogger(save_dir=str(log_dir), name="pretrain"),
        precision=precision,
        strategy="ddp_find_unused_parameters_true",
        sync_batchnorm=accelerator != "cpu",  # Sync batchnorm is not supported on CPU.
        num_sanity_val_steps=0,
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=ckpt_path,
    )
    for metric in ["val_online_cls_top1", "val_online_cls_top5"]:
        print_rank_zero(f"max {metric}: {max(metric_callback.val_metrics[metric])}")


def eval_metrics_to_markdown(metrics: Dict[str, Dict[str, float]]) -> str:
    EVAL_NAME_COLUMN_NAME = "Eval Name"
    METRIC_COLUMN_NAME = "Metric Name"
    VALUE_COLUMN_NAME = "Value"

    eval_name_max_len = max(
        len(eval_name) for eval_name in list(metrics.keys()) + [EVAL_NAME_COLUMN_NAME]
    )
    metric_name_max_len = max(
        len(metric_name)
        for metric_dict in metrics.values()
        for metric_name in list(metric_dict.keys()) + [METRIC_COLUMN_NAME]
    )
    value_max_len = max(
        len(metric_value)
        for metric_dict in metrics.values()
        for metric_value in list(f"{value:.3f}" for value in metric_dict.values())
        + [VALUE_COLUMN_NAME]
    )

    header = f"| {EVAL_NAME_COLUMN_NAME.ljust(eval_name_max_len)} | {METRIC_COLUMN_NAME.ljust(metric_name_max_len)} | {VALUE_COLUMN_NAME.ljust(value_max_len)} |"
    separator = f"|:{'-' * (eval_name_max_len)}:|:{'-' * (metric_name_max_len)}:|:{'-' * (value_max_len)}:|"

    lines = [header, separator] + [
        f"| {eval_name.ljust(eval_name_max_len)} | {metric_name.ljust(metric_name_max_len)} | {f'{metric_value:.4f}'.ljust(value_max_len)} |"
        for eval_name, metric_dict in metrics.items()
        for metric_name, metric_value in metric_dict.items()
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))

