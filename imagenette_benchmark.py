# -*- coding: utf-8 -*-
"""
Note that this benchmark also supports a multi-GPU setup. If you run it on
a system with multiple GPUs make sure that you kill all the processes when
killing the application. Due to the way we setup this benchmark the distributed
processes might continue the benchmark if one of the nodes is killed.
If you know how to fix this don't hesitate to create an issue or PR :)
You can download the ImageNette dataset from here: https://github.com/fastai/imagenette

Code has been tested on a A6000 GPU with 48GBytes of memory.

Code to reproduce the benchmark results:

Results (4.5.2023):
-------------------------------------------------------------------------------------------------
| Model            | Batch Size | Epochs |  KNN Top1 Val Accuracy |       Time | Peak GPU Usage |
-------------------------------------------------------------------------------------------------
| BarlowTwins      |        256 |    200 |                  0.651 |   85.0 Min |      4.0 GByte |
| BYOL             |        256 |    200 |                  0.705 |   54.4 Min |      4.3 GByte |
| DCL              |        256 |    200 |                  0.809 |   48.7 Min |      3.7 GByte |
| DCLW             |        256 |    200 |                  0.783 |   47.3 Min |      3.7 GByte |
| DINO (Res18)     |        256 |    200 |                  0.873 |   75.4 Min |      6.6 GByte |
| FastSiam         |        256 |    200 |                  0.779 |   88.2 Min |      7.3 GByte |
| MAE (ViT-S)      |        256 |    200 |                  0.454 |   62.0 Min |      4.4 GByte |
| MSN (ViT-S)      |        256 |    200 |                  0.713 |  127.0 Min |     14.7 GByte |
| Moco             |        256 |    200 |                  0.786 |   57.5 Min |      4.3 GByte |
| NNCLR            |        256 |    200 |                  0.809 |   51.5 Min |      3.8 GByte |
| PMSN (ViT-S)     |        256 |    200 |                  0.705 |  126.9 Min |     14.7 GByte |
| SimCLR           |        256 |    200 |                  0.835 |   49.7 Min |      3.7 GByte |
| SimMIM (ViT-B32) |        256 |    200 |                  0.315 |  115.5 Min |      9.7 GByte |
| SimSiam          |        256 |    200 |                  0.752 |   58.2 Min |      3.9 GByte |
| SwaV             |        256 |    200 |                  0.861 |   73.3 Min |      6.4 GByte |
| SwaVQueue        |        256 |    200 |                  0.827 |   72.6 Min |      6.4 GByte |
| SMoG             |        256 |    200 |                  0.663 |   58.7 Min |      2.6 GByte |
| TiCo             |        256 |    200 |                  0.742 |   45.6 Min |      2.5 GByte |
| VICReg           |        256 |    200 |                  0.763 |   53.2 Min |      4.0 GByte |
| VICRegL          |        256 |    200 |                  0.689 |   56.7 Min |      4.0 GByte |
-------------------------------------------------------------------------------------------------
| BarlowTwins      |        256 |    800 |                  0.852 |  298.5 Min |      4.0 GByte |
| BYOL             |        256 |    800 |                  0.887 |  214.8 Min |      4.3 GByte |
| DCL              |        256 |    800 |                  0.861 |  189.1 Min |      3.7 GByte |
| DCLW             |        256 |    800 |                  0.865 |  192.2 Min |      3.7 GByte |
| DINO (Res18)     |        256 |    800 |                  0.888 |  312.3 Min |      6.6 GByte |
| FastSiam         |        256 |    800 |                  0.873 |  299.6 Min |      7.3 GByte |
| MAE (ViT-S)      |        256 |    800 |                  0.610 |  248.2 Min |      4.4 GByte |
| MSN (ViT-S)      |        256 |    800 |                  0.828 |  515.5 Min |     14.7 GByte |
| Moco             |        256 |    800 |                  0.874 |  231.7 Min |      4.3 GByte |
| NNCLR            |        256 |    800 |                  0.884 |  212.5 Min |      3.8 GByte |
| PMSN (ViT-S)     |        256 |    800 |                  0.822 |  505.8 Min |     14.7 GByte |
| SimCLR           |        256 |    800 |                  0.889 |  193.5 Min |      3.7 GByte |
| SimMIM (ViT-B32) |        256 |    800 |                  0.343 |  446.5 Min |      9.7 GByte |
| SimSiam          |        256 |    800 |                  0.872 |  206.4 Min |      3.9 GByte |
| SwaV             |        256 |    800 |                  0.902 |  283.2 Min |      6.4 GByte |
| SwaVQueue        |        256 |    800 |                  0.890 |  282.7 Min |      6.4 GByte |
| SMoG             |        256 |    800 |                  0.788 |  232.1 Min |      2.6 GByte |
| TiCo             |        256 |    800 |                  0.856 |  177.8 Min |      2.5 GByte |
| VICReg           |        256 |    800 |                  0.845 |  205.6 Min |      4.0 GByte |
| VICRegL          |        256 |    800 |                  0.778 |  218.7 Min |      4.0 GByte |
-------------------------------------------------------------------------------------------------

"""
import copy
import os
import sys
import time

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
from pytorch_lightning.loggers import TensorBoardLogger

from lightly.data import LightlyDataset
from lightly.loss import (
    NegativeCosineSimilarity,
)

from lightly.transforms import (
    FastSiamTransform,
)
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.utils import scheduler
from lightly.utils.benchmarking import BenchmarkModule
from lightly.utils.lars import LARS
from lightly.utils.debug import std_of_l2_normalized
from lightly.models._momentum import _do_momentum_update

logs_root_dir = os.path.join(os.getcwd(), "benchmark_logs")

# Random Generator
rng = np.random.default_rng()

num_workers = 12

# set max_epochs to 800 for long run (takes around 10h on a single V100)
max_epochs = 800
knn_k = 200
knn_t = 0.1
classes = 10
input_size = 128

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
num_views = 5
pseudo_batch_size = 256
batch_size = 256
accumulate_grad_batches = pseudo_batch_size // batch_size
# lr_factor = (pseudo_batch_size/2*num_views) / 256  # scales the learning rate linearly with batch size
lr_factor = pseudo_batch_size / 256  # scales the learning rate linearly with batch size

# Number of devices and hardware to use for training.
devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
accelerator = "gpu" if torch.cuda.is_available() else "cpu"

# Set precison to high to increase performance
torch.set_float32_matmul_precision('high')

if distributed:
    strategy = "ddp"
    # reduce batch size for distributed training
    batch_size = batch_size // devices
else:
    strategy = "auto"  # Set to "auto" if using PyTorch Lightning >= 2.0
    # limit to single device if not using distributed training
    devices = min(devices, 1)

# The dataset structure should be like this:
path_to_train = "/media/tvanzyl/data/imagenette2-160/train/"
path_to_test = "/media/tvanzyl/data/imagenette2-160/val/"

# Use FastSiam augmentations
simsimp_transform = FastSiamTransform(
    num_views=num_views,
    input_size=input_size)

normalize_transform = torchvision.transforms.Normalize(
    mean=IMAGENET_NORMALIZE["mean"],
    std=IMAGENET_NORMALIZE["std"],
)

# No additional augmentations for the test set
test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(input_size),
        torchvision.transforms.CenterCrop(128),
        torchvision.transforms.ToTensor(),
        normalize_transform,
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


def get_data_loaders(batch_size: int, dataset_train_ssl):
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

class SimSimPModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        self.automatic_optimization = False
        self.fastforward = True
        self.layernorm = False
        self.drift = False
        # create a ResNet backbone and remove the classification head        
        self.ens_size = num_views
        resnet = torchvision.models.resnet18()        
        emb_width = list(resnet.children())[-1].in_features        
        self.upd_width = upd_width = 1024
        self.prd_width = prd_width = 512

        self.backbone = nn.Sequential(*list(resnet.children())[:-1],
                                    #   nn.Flatten(start_dim=1),
                                    #   nn.BatchNorm1d(emb_width, affine=False),
                                      )
        projection_head = []
        projection_head_ = nn.Sequential(
                nn.Identity(),
                # nn.LayerNorm((num_views, emb_width), elementwise_affine=False),
                nn.Linear(emb_width, upd_width),
                nn.BatchNorm1d(upd_width),
                nn.ReLU(inplace=True),
                nn.Linear(upd_width, upd_width),
                nn.BatchNorm1d(upd_width, affine=False),
            )
        for i in range(self.ens_size):            
            projection_head.append(
                projection_head_
            )
        self.projection_head = nn.ModuleList(projection_head)
        prediction_head = []                
        prediction_head_ = nn.Sequential(      
                nn.ReLU(inplace=True),
                nn.Linear(upd_width, prd_width, False),
            )
        for i in range(self.ens_size):                        
            prediction_head.append(
                prediction_head_
            )
        self.prediction_head = nn.ModuleList(prediction_head)
        merge_head = []    
        self.rand_proj = nn.Linear(upd_width, prd_width)
        nn.init.orthogonal_(self.rand_proj.weight)
        merge_head_ = nn.Sequential(
                self.rand_proj
            )        
        for i in range(self.ens_size):                        
            merge_head.append(
                merge_head_
            )
        self.merge_head = nn.ModuleList(merge_head)
        self.criterion = NegativeCosineSimilarity()

    # def forward_(self, x, i):
    #     f_ = self.backbone( x[i] ).flatten(start_dim=1)
    #     g_ = self.projection_head[i]( f_ )        
    #     p_ = self.prediction_head[i]( g_ )
    #     return p_
    
    # def forward(self, x):
    #     g, z = [], []
    #     with torch.no_grad():
    #         for i in range(self.ens_size):
    #             f_ = self.backbone( x[i] ).flatten(start_dim=1)
    #             g_ = self.projection_head[i]( f_ )
    #             g.append( g_.detach() )
    #         for i in range(self.ens_size):
    #             e_ = torch.concat([g[j] for j in range(self.ens_size) if j != i], dim=1)
    #             # e_ = torch.stack([g[j] for j in range(self.ens_size) if j != i], dim=2).mean(dim=2)
    #             z_  = self.merge_head[i]( e_ )
    #             z.append( z_ )
    #     return z

    # def ffforward(self, x):
    #     p, g, e, z, f = [], [], [], [], []
    #     for i in range(self.ens_size):
    #         f_ = self.backbone( x[i] ).flatten(start_dim=1)
    #         f.append( f_ )
    #     f__ = torch.stack([f[i] for i in range(self.ens_size)], dim=1)
    #     g__ = self.projection_head[0]( f__ )
    #     for i in range(self.ens_size):            
    #         g_ = g__[:,i]
    #         p_ = self.prediction_head[0]( g_ )
    #         g.append( g_ )
    #         p.append( p_ )
    #         with torch.no_grad():
    #             e_ = self.merge_head[0]( g_.detach() )
    #         e.append( e_ )
    #     for i in range(self.ens_size):
    #         z_ = torch.stack([e[j] for j in range(self.ens_size) if j != i], dim=2).mean(dim=2)
    #         z.append( z_ )            
    #     return p, z, g

    def fforward(self, x):
        p, g, e, z = [], [], [], []
        for i in range(self.ens_size):
            f_ = self.backbone( x[i] ).flatten(start_dim=1)
            g_ = self.projection_head[i]( f_ )
            g.append( g_.detach() )
            p_ = self.prediction_head[i]( g_ )
            p.append( p_ )
            with torch.no_grad():
                e_ = self.merge_head[i]( g_.detach() )
            e.append( e_ )
        # with torch.no_grad():
        #     for i in range(self.ens_size):
                # e_ = torch.concat([g[j] for j in range(self.ens_size) if j != i], dim=1)
                # e_ = torch.stack([g[j] for j in range(self.ens_size) if j != i], dim=2).mean(dim=2)
                # z_  = self.merge_head[i]( e_ )
                # z.append( z_ )        
        for i in range(self.ens_size):
            z_ = torch.stack([e[j] for j in range(self.ens_size) if j != i], dim=2).mean(dim=2)
            z.append( z_ )
        return p, z, g

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()                
        sch = self.lr_schedulers()
        x, _, _ = batch        
        loss_tot_l = 0

        if self.fastforward and self.layernorm:            
            p, z, g = self.ffforward( x )
        elif self.fastforward:
            p, z, g = self.fforward( x )
        else:
            z = self.forward( x )
        
        for xi in range(self.ens_size):
            p_ = p[xi] if self.fastforward else self.forward_(x, xi)            
            z_ = z[xi]
            #increase diversity with abs()
            loss_l = self.criterion( p_, z_  ) / self.ens_size
            self.manual_backward( loss_l )#, retain_graph=True)
            loss_tot_l += loss_l.detach()

        with torch.no_grad():
            f_ = self.backbone(x[xi]).flatten(start_dim=1)
        g_ = g[xi]
        self.log("f_", std_of_l2_normalized(f_),   prog_bar=True)
        self.log("g_", std_of_l2_normalized(g_),   prog_bar=True)
        self.log("z_", std_of_l2_normalized(z_),   prog_bar=True)        
        self.log("p_", std_of_l2_normalized(p_),   prog_bar=True)

        if self.drift:
            with torch.no_grad():
                rand_proj = nn.Linear(self.upd_width, self.prd_width)
                nn.init.orthogonal_(rand_proj)
                _do_momentum_update(self.rand_proj.parameters(), 
                                    rand_proj.parameters(),
                                    0.999)

        if self.trainer.is_last_batch:
            opt.step()
            opt.zero_grad()
            sch.step()            
            # print(f_[0, :8].detach().tolist())
            # print(f_[1, :8].detach().tolist())
            # print("f---")
            # print(g_[0, :8].detach().tolist())
            # print(g_[1, :8].detach().tolist())
            # print("g---")
            # print(p_[0, :8].detach().tolist())
            # print(p_[1, :8].detach().tolist())
            # print("p---")
            # print(z_[0, :8].detach().tolist())
            # print(z_[1, :8].detach().tolist())
            # print("z---")
        elif (batch_idx + 1) % accumulate_grad_batches == 0:
            opt.step()
            opt.zero_grad()
        
        self.log("pred_l", loss_tot_l,   prog_bar=True)

    def configure_optimizers(self):
        optim = torch.optim.SGD(    
            self.parameters(),
            lr=6e-2*lr_factor,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]

# class SimSiamModel(BenchmarkModule):
#     def __init__(self, dataloader_kNN, num_classes):
#         super().__init__(dataloader_kNN, num_classes)
#         # create a ResNet backbone and remove the classification head
#         resnet = torchvision.models.resnet18()
#         feature_dim = list(resnet.children())[-1].in_features
#         self.backbone = nn.Sequential(*list(resnet.children())[:-1])
#         self.projection_head = heads.SimSiamProjectionHead(feature_dim, 2048, 2048)
#         self.prediction_head = heads.SimSiamPredictionHead(2048, 512, 2048)
#         self.criterion = NegativeCosineSimilarity()

#     def forward(self, x):
#         f = self.backbone(x).flatten(start_dim=1)
#         z = self.projection_head(f)
#         p = self.prediction_head(z)
#         z = z.detach()
#         return z, p

#     def training_step(self, batch, batch_idx):
#         (x0, x1), _, _ = batch
#         z0, p0 = self.forward(x0)
#         z1, p1 = self.forward(x1)
#         loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
#         self.log("train_loss_ssl", loss)
#         return loss

#     def configure_optimizers(self):
#         optim = torch.optim.SGD(
#             self.parameters(),
#             lr=6e-2,  # no lr-scaling, results in better training stability
#             momentum=0.9,
#             weight_decay=5e-4,
#         )
#         cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
#         return [optim], [cosine_scheduler]

models = [
    SimSimPModel,
]

bench_results = dict()

experiment_version = None
# loop through configurations and train models
for BenchmarkModel in models:
    runs = []
    model_name = BenchmarkModel.__name__.replace("Model", "")
    for seed in range(n_runs):
        # pl.seed_everything(seed)
        dataset_train_ssl = create_dataset_train_ssl(BenchmarkModel)
        dataloader_train_ssl, dataloader_train_kNN, dataloader_test = get_data_loaders(
            batch_size=batch_size, dataset_train_ssl=dataset_train_ssl
        )
        benchmark_model = BenchmarkModel(dataloader_train_kNN, classes)

        # Save logs to: {CWD}/benchmark_logs/cifar10/{experiment_version}/{model_name}/
        # If multiple runs are specified a subdirectory for each run is created.
        sub_dir = model_name if n_runs <= 1 else f"{model_name}/run{seed}"
        logger = TensorBoardLogger(
            save_dir=os.path.join(logs_root_dir, "imagenette"),
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


