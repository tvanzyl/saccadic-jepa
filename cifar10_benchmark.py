# -*- coding: utf-8 -*-
"""
Benchmark Results

Updated: 27.03.2023 (42a6a924b1b6d5b6cc89a6b2a0a0942cc4af93ab)

------------------------------------------------------------------------------------------
| Model         | Batch Size | Epochs |  KNN Test Accuracy |       Time | Peak GPU Usage |
------------------------------------------------------------------------------------------
| BarlowTwins   |        128 |    200 |              0.842 |  375.9 Min |      1.7 GByte |
| BYOL          |        128 |    200 |              0.869 |  121.9 Min |      1.6 GByte |
| DCL           |        128 |    200 |              0.844 |  102.2 Min |      1.5 GByte |
| DCLW          |        128 |    200 |              0.833 |  100.4 Min |      1.5 GByte |
| DINO          |        128 |    200 |              0.840 |  120.3 Min |      1.6 GByte |
| FastSiam      |        128 |    200 |              0.906 |  164.0 Min |      2.7 GByte |
| Moco          |        128 |    200 |              0.838 |  128.8 Min |      1.7 GByte |
| NNCLR         |        128 |    200 |              0.834 |  101.5 Min |      1.5 GByte |
| SimCLR        |        128 |    200 |              0.847 |   97.7 Min |      1.5 GByte |
| SimSiam       |        128 |    200 |              0.819 |   97.3 Min |      1.6 GByte |
| SwaV          |        128 |    200 |              0.812 |   99.6 Min |      1.5 GByte |
| SMoG          |        128 |    200 |              0.743 |  192.2 Min |      1.2 GByte |
------------------------------------------------------------------------------------------
| BarlowTwins   |        512 |    200 |              0.819 |  153.3 Min |      5.1 GByte |
| BYOL          |        512 |    200 |              0.868 |  108.3 Min |      5.6 GByte |
| DCL           |        512 |    200 |              0.840 |   88.2 Min |      4.9 GByte |
| DCLW          |        512 |    200 |              0.824 |   87.9 Min |      4.9 GByte |
| DINO          |        512 |    200 |              0.813 |  108.6 Min |      5.0 GByte |
| FastSiam      |        512 |    200 |              0.788 |  146.9 Min |      9.5 GByte |
| Moco (*)      |        512 |    200 |              0.847 |  112.2 Min |      5.6 GByte |
| NNCLR (*)     |        512 |    200 |              0.815 |   88.1 Min |      5.0 GByte |
| SimCLR        |        512 |    200 |              0.848 |   87.1 Min |      4.9 GByte |
| SimSiam       |        512 |    200 |              0.764 |   87.8 Min |      5.0 GByte |
| SwaV          |        512 |    200 |              0.842 |   88.7 Min |      4.9 GByte |
| SMoG          |        512 |    200 |              0.686 |  110.0 Min |      3.4 GByte |
------------------------------------------------------------------------------------------
| BarlowTwins   |        512 |    800 |              0.859 |  517.5 Min |      7.9 GByte |
| BYOL          |        512 |    800 |              0.910 |  400.9 Min |      5.4 GByte |
| DCL           |        512 |    800 |              0.874 |  334.6 Min |      4.9 GByte |
| DCLW          |        512 |    800 |              0.871 |  333.3 Min |      4.9 GByte |
| DINO          |        512 |    800 |              0.848 |  405.2 Min |      5.0 GByte |
| FastSiam      |        512 |    800 |              0.902 |  582.0 Min |      9.5 GByte |
| Moco (*)      |        512 |    800 |              0.899 |  417.8 Min |      5.4 GByte |
| NNCLR (*)     |        512 |    800 |              0.892 |  335.0 Min |      5.0 GByte |
| SimCLR        |        512 |    800 |              0.879 |  331.1 Min |      4.9 GByte |
| SimSiam       |        512 |    800 |              0.904 |  333.7 Min |      5.1 GByte |
| SwaV          |        512 |    800 |              0.884 |  330.5 Min |      5.0 GByte |
| SMoG          |        512 |    800 |              0.800 |  415.6 Min |      3.2 GByte |
------------------------------------------------------------------------------------------

(*): Increased size of memory bank from 4096 to 8192 to avoid too quickly 
changing memory bank due to larger batch size.

The benchmarks were created on a single NVIDIA RTX A6000.

Note that this benchmark also supports a multi-GPU setup. If you run it on
a system with multiple GPUs make sure that you kill all the processes when
killing the application. Due to the way we setup this benchmark the distributed
processes might continue the benchmark if one of the nodes is killed.
If you know how to fix this don't hesitate to create an issue or PR :)

"""
import copy
import os
import time

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F

import torchvision
from pytorch_lightning.loggers import TensorBoardLogger

from lightly.data import LightlyDataset
from lightly.loss import (
    BarlowTwinsLoss,
    DCLLoss,
    DCLWLoss,
    DINOLoss,
    NegativeCosineSimilarity,
    NTXentLoss,
    SwaVLoss,
)
from lightly.models import ResNetGenerator, modules, utils
from lightly.models.modules import heads, memory_bank
from lightly.transforms import (
    BYOLTransform,
    BYOLView1Transform,
    BYOLView2Transform,
    DINOTransform,
    FastSiamTransform,
    SimCLRTransform,
    SimSiamTransform,
    SMoGTransform,
    SwaVTransform,
    MAETransform,
)
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.utils.benchmarking import BenchmarkModule

from lightly.utils.debug import std_of_l2_normalized

from SimplRSiam import L2NormalizationLayer

logs_root_dir = os.path.join(os.getcwd(), "benchmark_logs")

# Random Generator
rng = np.random.default_rng()

# Trade-off precision for performance.
torch.set_float32_matmul_precision('high')

# set max_epochs to 800 for long run (takes around 10h on a single V100)
max_epochs = 200
num_workers = 8
knn_k = 200
knn_t = 0.1
classes = 10

# Set to True to enable Distributed Data Parallel training.
distributed = False

# Set to True to enable Synchronized Batch Norm (requires distributed=True).
# If enabled the batch norm is calculated over all gpus, otherwise the batch
# norm is only calculated from samples on the same gpu.
sync_batchnorm = False

# Set to True to gather features from all gpus before calculating
# the loss (requires distributed=True).
# If enabled then the loss on every gpu is calculated with features from all
# gpus, otherwise only features from the same gpu are used.
gather_distributed = False

# benchmark
n_runs = 1  # optional, increase to create multiple runs and report mean + std
pseudo_batch_size = 64
batch_size = pseudo_batch_size
accumulate_grad_batches = pseudo_batch_size // batch_size
lr_factor = pseudo_batch_size / 128  # scales the learning rate linearly with batch size

# Number of devices and hardware to use for training.
devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
accelerator = "gpu" if torch.cuda.is_available() else "cpu"

if distributed:
    strategy = "ddp"
    # reduce batch size for distributed training
    batch_size = batch_size // devices
else:
    strategy = "auto"  # Set to "auto" if using PyTorch Lightning >= 2.0
    # limit to single device if not using distributed training
    devices = min(devices, 1)

# Adapted from our MoCo Tutorial on CIFAR-10
#
# Replace the path with the location of your CIFAR-10 dataset.
# We assume we have a train folder with subfolders
# for each class and .png images inside.
#
# You can download `CIFAR-10 in folders from kaggle
# <https://www.kaggle.com/swaroopkml/cifar10-pngs-in-folders>`_.

# The dataset structure should be like this:
# cifar10/train/
#  L airplane/
#    L 10008_airplane.png
#    L ...
#  L automobile/
#  L bird/
#  L cat/
#  L deer/
#  L dog/
#  L frog/
#  L horse/
#  L ship/
#  L truck/
path_to_train = "/media/tvanzyl/data/cifar10/train/"
path_to_test = "/media/tvanzyl/data/cifar10/test/"

# Use BYOL augmentations
byol_transform = BYOLTransform(
    view_1_transform=BYOLView1Transform(input_size=32, gaussian_blur=0.0),
    view_2_transform=BYOLView2Transform(input_size=32, gaussian_blur=0.0),
)

# Use SimCLR augmentations
simclr_transform = SimCLRTransform(
    input_size=32,
    cj_strength=0.5,
    gaussian_blur=0.0,
)

# Use SimSiam augmentations
simsiam_transform = SimSiamTransform(
    input_size=32,
    gaussian_blur=0.0,
)

input_size=32
# Use Multi-Crop augmentations https://arxiv.org/html/2403.05726v1#bib.bib7
num_local_views = {32:0,64:6,96:6,128:6,224:6}[input_size]
num_views = 2 + num_local_views
simsimp_transform = {
32:DINOTransform(global_crop_size=32,
                 global_crop_scale=(0.14, 1.0),
                 n_local_views=0,
                 gaussian_blur=(0, 0, 0),
                ),
64:DINOTransform(global_crop_size=64,
                 global_crop_scale=(0.25, 1.0),
                 local_crop_size=32,
                 local_crop_scale=(0.14, 0.25),
                #  gaussian_blur=(0, 0, 0),
                ),
96:DINOTransform(global_crop_size=96,
                 global_crop_scale=(0.25, 1.0),
                 local_crop_size=48,
                 local_crop_scale=(0.14, 0.25),
                ),
128:DINOTransform(global_crop_size=128,
                  global_crop_scale=(0.25, 1.0),
                  local_crop_size=64,
                  local_crop_scale=(0.08, 0.25),
                ),
244:DINOTransform(global_crop_size=224,
                  global_crop_scale=(0.25, 1.0),
                  local_crop_scale =(0.08, 0.25),
                ),
}[input_size]
# num_views=2
# simsimp_transform = BYOLTransform(
#     view_1_transform=BYOLView1Transform(input_size=32, gaussian_blur=0.0, min_scale=0.08),
#     view_2_transform=BYOLView2Transform(input_size=32, gaussian_blur=0.0, min_scale=0.08),
# )

# Multi crop augmentation for FastSiam
fast_siam_transform = FastSiamTransform(input_size=32, gaussian_blur=0.0)

# Multi crop augmentation for SwAV, additionally, disable blur for cifar10
swav_transform = SwaVTransform(
    crop_sizes=[32,32],
    crop_counts=[2,2],  # 2 crops @ 32x32px
    crop_min_scales=[0.14,0.14],
    cj_strength=0.5,
    gaussian_blur=0,
)

# Multi crop augmentation for DINO, additionally, disable blur for cifar10
dino_transform = DINOTransform(
    global_crop_size=32,
    n_local_views=0,
    cj_strength=0.5,
    gaussian_blur=(0, 0, 0),
)

# Two crops for SMoG
smog_transform = SMoGTransform(
    crop_sizes=(32, 32),
    crop_counts=(1, 1),
    cj_strength=0.5,
    gaussian_blur_probs=(0.0, 0.0),
    crop_min_scales=(0.2, 0.2),
    crop_max_scales=(1.0, 1.0),
)

simmim_transform = MAETransform(32)

# No additional augmentations for the test set
test_transforms = torchvision.transforms.Compose(
    [   
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=IMAGENET_NORMALIZE["mean"],
            std=IMAGENET_NORMALIZE["std"],
        ),
    ]
)

# we use test transformations for getting the feature for kNN on train data
dataset_train_kNN = LightlyDataset(input_dir=path_to_train, transform=test_transforms)

dataset_test = LightlyDataset(input_dir=path_to_test, transform=test_transforms)

def create_dataset_train_ssl(model):
    """Helper method to apply the correct transform for ssl.

    Args:
        model:
            Model class for which to select the transform.
    """
    model_to_transform = {        
        SimSimPModel: simsimp_transform,

    }
    transform = model_to_transform[model]
    return LightlyDataset(input_dir=path_to_train, transform=transform)


def get_data_loaders(batch_size: int, dataset_train_ssl, collator):
    """Helper method to create dataloaders for ssl, kNN train and kNN test.

    Args:
        batch_size: Desired batch size for all dataloaders.
    """
    dataloader_train_ssl = torch.utils.data.DataLoader(
        dataset_train_ssl,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        collate_fn=collator,
    )

    dataloader_train_kNN = torch.utils.data.DataLoader(
        dataset_train_kNN,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    return dataloader_train_ssl, dataloader_train_kNN, dataloader_test

from lightly.models.modules import MaskedVisionTransformerTorchvision

class SimSimPModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        self.automatic_optimization = False        
        # create a ResNet backbone and remove the classification head
        emb_width = 512
        resnet = ResNetGenerator("resnet-18", width=emb_width/512.0)
        
        self.ens_size = num_views
        self.upd_width = upd_width = 512
        self.prd_width = prd_width = 1024

        self.backbone = nn.Sequential(
                            *list(resnet.children())[:-1],
                            nn.AdaptiveAvgPool2d(1)
                        )    
            
        self.projection_head = nn.Sequential(
                nn.Linear(emb_width, upd_width),
                nn.BatchNorm1d(upd_width),
                nn.ReLU(inplace=True),
                nn.Linear(upd_width, prd_width),
                L2NormalizationLayer(),
                nn.BatchNorm1d(prd_width, affine=False),                
            )
        
        self.rand_proj_q = nn.Linear(prd_width, emb_width, False)
        self.prediction_head = nn.Sequential(
                nn.ReLU(inplace=True),
                self.rand_proj_q,
            )        
        
        self.rand_proj_n = nn.Linear(prd_width, emb_width) 
        self.rand_proj_n.weight.data = self.rand_proj_q.weight.data
        self.merge_head = self.rand_proj_n

        self.criterion = NegativeCosineSimilarity()

    def forward(self, x):
        p, g, e, z = [], [], [], []
        #Pass Through Each Global Seperate
        for i in range(2):
            f_ = self.backbone( x[i] ).flatten(start_dim=1)
            g_ = self.projection_head( f_ )
            g.append( g_.detach() )
            p_ = self.prediction_head( g_ )
            p.append( p_ )
        with torch.no_grad():
            # e_ = torch.stack([g[j] for j in range(self.ens_size) if j != i], dim=1).mean(dim=1)            
            # z1_ = self.merge_head( g[1] )
            # z.append( z1_)
            # z0_ = self.merge_head( g[0] )
            # z.append( z0_)
            e_ = torch.stack([g[0], g[1]], dim=1).mean(dim=1)
            z_ = self.merge_head( e_ )
        for i in range(self.ens_size):
            z.append( z_ )

        #Pass Through The Locals Together
        if self.ens_size > 2:
            x_ = torch.cat(x[2:])
            f_ = self.backbone( x_ ).flatten(start_dim=1)
            g_ = self.projection_head( f_ )
            # g.append( g_.detach().chunk(self.ens_size-2) )
            p_ = self.prediction_head( g_ )
            p.append( p_.chunk(self.ens_size-2) )

        return f_.detach(), p, z, g_.detach()
    
    def training_step(self, batch, batch_idx):
        opt = self.optimizers()                
        sch = self.lr_schedulers()
        x, _, _ = batch        
        loss_tot_l = 0

        f_, p, z, g_ = self.forward( x )
        
        loss_l = 0
        for xi in range(self.ens_size):
            p_ = p[xi]
            z_ = z[xi]            
            loss_l += self.criterion( p_, z_ ) / self.ens_size
        loss_tot_l += loss_l.detach()
        self.manual_backward( loss_l )

        self.log("f_", std_of_l2_normalized(f_))
        self.log("g_", std_of_l2_normalized(g_))
        self.log("z_", std_of_l2_normalized(z_))
        self.log("p_", std_of_l2_normalized(p_))

        if self.trainer.is_last_batch:
            opt.step()
            opt.zero_grad()
            sch.step()
        elif (batch_idx + 1) % accumulate_grad_batches == 0:
            opt.step()
            opt.zero_grad()
        
        self.log("pred_l", loss_tot_l,   prog_bar=True)

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(),
            lr=6e-2, #*lr_factor,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim] , [scheduler]


from lightly.utils.benchmarking.knn import knn_predict
import torch.distributed as dist
from lightly.utils.dist import gather as lightly_gather

class BenchmarkModuleP(BenchmarkModule):
    def __init__(
        self,
        dataloader_kNN,
        num_classes,
        knn_k= 200,
        knn_t= 0.1,
    ):
        super().__init__(dataloader_kNN,
                         num_classes,
                         knn_k,
                         knn_t,)
        self.max_accuracy_m = 0.0
        self._train_features_m = None
        self._train_targets_m = None
        self._val_predicted_labels_m = []
        self._val_targets_m = []

    def on_validation_epoch_start(self):   
        utils.update_momentum(self.backbone, self.backbone_momentum, m=0.99)
        utils.update_momentum(
            self.projection_head, self.projection_head_momentum, m=0.99
        )     
        super().on_validation_epoch_start()        
        train_features = []
        train_targets = []
        with torch.no_grad():            
            for data in self.dataloader_kNN:                
                img, target, _ = data
                img = img.to(self.device)
                target = target.to(self.device)
                feature = self.backbone_momentum(img).squeeze()
                feature = F.normalize(feature, dim=1)
                train_features.append(feature)
                train_targets.append(target)
        self._train_features_m = torch.cat(train_features, dim=0).t().contiguous()
        self._train_targets_m = torch.cat(train_targets, dim=0).t().contiguous()

    def validation_step(self, batch, batch_idx):        
        super().validation_step(batch, batch_idx)
        # we can only do kNN predictions once we have a feature bank
        if self._train_features_m is not None and self._train_targets_m is not None:            
            images, targets, _ = batch
            feature = self.backbone_momentum(images).squeeze()
            feature = F.normalize(feature, dim=1)
            predicted_labels = knn_predict(
                feature,
                self._train_features_m,
                self._train_targets_m,
                self.num_classes,
                self.knn_k,
                self.knn_t,
            )

            if dist.is_initialized() and dist.get_world_size() > 0:
                # gather predictions and targets from all processes

                predicted_labels = torch.cat(lightly_gather(predicted_labels), dim=0)
                targets = torch.cat(lightly_gather(targets), dim=0)

            self._val_predicted_labels_m.append(predicted_labels.cpu())
            self._val_targets_m.append(targets.cpu())

    def on_validation_epoch_end(self):        
        super().on_validation_epoch_end()
        if self._val_predicted_labels_m and self._val_targets_m:            
            predicted_labels = torch.cat(self._val_predicted_labels_m, dim=0)
            targets = torch.cat(self._val_targets_m, dim=0)
            top1 = (predicted_labels[:, 0] == targets).float().sum()
            acc = top1 / len(targets)
            if acc > self.max_accuracy_m:
                self.max_accuracy_m = float(acc.item())
            self.log("kNN_accu_mom", acc * 100.0, prog_bar=True)

        self._val_predicted_labels_m.clear()
        self._val_targets_m.clear()

from sklearn.cluster import KMeans

from typing import Callable, List, Union

def repeat_interleave_batch(
    self, x: torch.Tensor, B: int, repeat: int
) -> torch.Tensor:
    """Repeat and interleave the input tensor."""
    N = len(x) // B
    x = torch.cat(
        [
            torch.cat([x[i * B : (i + 1) * B] for _ in range(repeat)], dim=0)
            for i in range(N)
        ],
        dim=0,
    )
    return x

def apply_masks(
    self, x: torch.Tensor, masks: Union[torch.Tensor, List[torch.Tensor]]
) -> torch.Tensor:
    """
    From https://github.com/facebookresearch/ijepa/blob/main/src/masks/utils.py
    Apply masks to the input tensor.

    Args:
        x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
        masks: tensor or list of tensors containing indices of patches in [N] to keep
    Returns:
        tensor of shape [B, N', D] where N' is the number of patches to keep
    """
    if not isinstance(masks, list):
        masks = [masks]

    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    return torch.cat(all_x, dim=0)


models = [
    # BarlowTwinsModel,
    # BYOLModel,
    # DCL,
    # DCLW,
    # DINOModel,
    # MocoModel,
    # NNCLRModel,
    # SimCLRModel,
    # SimSiamModel,
    SimSimPModel,    
    # SwaVModel,
    # SMoGModel,
    # SimMIM,
]
bench_results = dict()

experiment_version = None
# loop through configurations and train models
for BenchmarkModel in models:
    runs = []
    model_name = BenchmarkModel.__name__.replace("Model", "")
    for seed in range(n_runs):
        pl.seed_everything(seed)               
        dataset_train_ssl = create_dataset_train_ssl(BenchmarkModel)
        dataloader_train_ssl, dataloader_train_kNN, dataloader_test = get_data_loaders(
            batch_size=batch_size, dataset_train_ssl=dataset_train_ssl, collator=None
        )
        benchmark_model = BenchmarkModel(dataloader_train_kNN, classes)

        # Save logs to: {CWD}/benchmark_logs/cifar10/{experiment_version}/{model_name}/
        # If multiple runs are specified a subdirectory for each run is created.
        sub_dir = model_name if n_runs <= 1 else f"{model_name}/run{seed}"
        logger = TensorBoardLogger(
            save_dir=os.path.join(logs_root_dir, "cifar10"),
            name="",
            sub_dir=sub_dir,
            version=experiment_version,
        )
        if experiment_version is None:
            # Save results of all models under same version directory
            experiment_version = logger.version
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(logger.log_dir, "checkpoints")
        )
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            devices=devices,
            accelerator=accelerator,
            default_root_dir=logs_root_dir,
            strategy=strategy,
            sync_batchnorm=sync_batchnorm,
            logger=logger,
            callbacks=[checkpoint_callback],            
        )
        start = time.time()
        trainer.fit(
            benchmark_model,
            train_dataloaders=dataloader_train_ssl,
            val_dataloaders=dataloader_test,
        )
        end = time.time()
        run = {
            "model": model_name,
            "batch_size": batch_size,
            "epochs": max_epochs,
            "max_accuracy": benchmark_model.max_accuracy,
            "runtime": end - start,
            "gpu_memory_usage": torch.cuda.max_memory_allocated(),
            "seed": seed,
        }
        runs.append(run)
        print(run)

        # delete model and trainer + free up cuda memory
        del benchmark_model
        del trainer
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    bench_results[model_name] = runs

# print results table
header = (
    f"| {'Model':<13} | {'Batch Size':>10} | {'Epochs':>6} "
    f"| {'KNN Test Accuracy':>18} | {'Time':>10} | {'Peak GPU Usage':>14} |"
)
print("-" * len(header))
print(header)
print("-" * len(header))
for model, results in bench_results.items():
    runtime = np.array([result["runtime"] for result in results])
    runtime = runtime.mean() / 60  # convert to min
    accuracy = np.array([result["max_accuracy"] for result in results])
    gpu_memory_usage = np.array([result["gpu_memory_usage"] for result in results])
    gpu_memory_usage = gpu_memory_usage.max() / (1024**3)  # convert to gbyte

    if len(accuracy) > 1:
        accuracy_msg = f"{accuracy.mean():>8.3f} +- {accuracy.std():>4.3f}"
    else:
        accuracy_msg = f"{accuracy.mean():>18.3f}"

    print(
        f"| {model:<13} | {batch_size:>10} | {max_epochs:>6} "
        f"| {accuracy_msg} | {runtime:>6.1f} Min "
        f"| {gpu_memory_usage:>8.1f} GByte |",
        flush=True,
    )
print("-" * len(header))


