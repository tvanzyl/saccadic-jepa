# -*- coding: utf-8 -*-
"""
Benchmark Results

Updated: 13.02.2023

"""
import copy
import os
import time

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import LambdaLR

from lightly.data import LightlyDataset
from lightly.loss import (
    NegativeCosineSimilarity,
)
from lightly.models import modules, utils
from lightly.models.modules import heads

from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.utils.benchmarking import BenchmarkModule
from lightly.utils.lars import LARS
from lightly.utils.scheduler import CosineWarmupScheduler
from lightly.utils.debug import std_of_l2_normalized
from lightly.models._momentum import _do_momentum_update
from lightly.models.utils import update_momentum
from lightly.utils.scheduler import cosine_schedule

from SimplRSiam import L2NormalizationLayer, SIMSIMPTRansform

logs_root_dir = os.path.join(os.getcwd(), "benchmark_logs")

# Trade-off precision for performance.
torch.set_float32_matmul_precision('high')

num_workers = 12

# set max_epochs to 800 for long run (takes around 10h on a single V100)
max_epochs = 200
knn_k = 20
knn_t = 0.1
classes = 200
input_size = 64

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
pseudo_batch_size = 1024
batch_size = pseudo_batch_size
accumulate_grad_batches = pseudo_batch_size // batch_size
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

path_to_train = "/media/tvanzyl/data/tiny-imagenet-200/train/"
path_to_test = "/media/tvanzyl/data/tiny-imagenet-200/val/"

# Use Multi-Crop augmentations https://arxiv.org/html/2403.05726v1#bib.bib7
num_local_views = {32:0,64:6,96:6,128:6,224:6}[input_size]
num_views = 2 + num_local_views
simsimp_transform = SIMSIMPTRansform[input_size]

# No additional augmentations for the test set
test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(input_size),
        torchvision.transforms.CenterCrop(64),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=IMAGENET_NORMALIZE["mean"],
            std=IMAGENET_NORMALIZE["std"],
        ),
    ]
)

dataset_train_ssl = LightlyDataset(input_dir=path_to_train)

# we use test transformations for getting the feature for kNN on train data
dataset_train_kNN = LightlyDataset(input_dir=path_to_train, transform=test_transforms)

dataset_test = LightlyDataset(input_dir=path_to_test, transform=test_transforms)

steps_per_epoch = len(LightlyDataset(input_dir=path_to_train)) // batch_size


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
        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        emb_width = list(resnet.children())[-1].in_features
        
        self.ens_size = num_views        
        self.upd_width = upd_width = 1024
        self.prd_width = prd_width = 512        

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.projection_head = nn.Sequential(
                nn.Linear(emb_width, upd_width),
                nn.BatchNorm1d(upd_width),
                nn.ReLU(inplace=True),
                nn.Linear(upd_width, prd_width),                
                L2NormalizationLayer(),
                nn.BatchNorm1d(prd_width, affine=False),
                nn.LeakyReLU(),
            )
        
        self.rand_proj_q = nn.Linear(prd_width, emb_width, False)        
        self.prediction_head = nn.Sequential(
                # nn.LeakyReLU(),
                self.rand_proj_q,
            )        
        self.rand_proj_n = nn.Linear(prd_width, emb_width) 
        self.rand_proj_n.weight.data = self.rand_proj_q.weight.data
        # nn.init.eye_(self.rand_proj_n.weight)
        # nn.init.orthogonal_(self.rand_proj_n.weight, gain=nn.init.calculate_gain('relu'))
        # self.rand_proj_n.bias.data[:] = 0.2
        self.merge_head =  nn.Sequential(
                # self.rand_proj_q,
                self.rand_proj_n,
            )        

        self.criterion = NegativeCosineSimilarity()

    def forward(self, x):
        p, g, e, z = [], [], [], []
        for i in range(self.ens_size):
            f_ = self.backbone( x[i] ).flatten(start_dim=1)            
            g_ = self.projection_head( f_ )
            g.append( g_.detach() )
            p_ = self.prediction_head( g_ )
            p.append( p_ )
        with torch.no_grad():
            e_ = torch.stack(g, dim=1).mean(dim=1)
            z_ = self.merge_head( e_ )
        for i in range(self.ens_size):
            z.append( z_ )
        return f_.detach(), p, z, g_.detach()

    # def forward(self, x):
    #     p, g, e, z = [], [], [], []
    #     for i in range(self.ens_size):
    #         f_ = self.backbone( x[i] ).flatten(start_dim=1)            
    #         g_ = self.projection_head( f_ )
    #         g.append( g_.detach() )
    #         p_ = self.prediction_head( g_ )
    #         p.append( p_ )
    #     with torch.no_grad():
    #         for i in range(self.ens_size):
    #             e_ = torch.stack([g[j] for j in range(self.ens_size) if j!=i], dim=1).mean(dim=1)
    #             z_ = self.merge_head( e_ )
    #             z.append( z_ )
        
    #     return f_.detach(), p, z, g_.detach()

    # def forward(self, x):
    #     p, g, e, z = [], [], [], []
    #     #Pass Through Each Global Seperate
    #     for i in range(2):
    #         f_ = self.backbone( x[i] ).flatten(start_dim=1)
    #         g_ = self.projection_head( f_ )
    #         g.append( g_.detach() )
    #         p_ = self.prediction_head( g_ )
    #         p.append( p_ )

    #     #Pass Through The Locals Together
    #     if self.ens_size > 2:
    #         x__ = torch.cat( x[2:] )
    #         f__ = self.backbone( x__ ).flatten(start_dim=1)
    #         g__ = self.projection_head( f__ )
    #         p__ = self.prediction_head( g__ )
    #         p.extend( p__.chunk(self.ens_size-2) )
        
    #     # Create The Teacher Weighted Equal To Globals and Locals
    #     with torch.no_grad():            
    #         # z0_ = self.merge_head( g[0] )
    #         # z1_ = self.merge_head( g[1] )
    #         e_ = torch.stack(g, dim=1).mean(dim=1)
    #         zg_ = self.merge_head( e_ )
    #         e__ = g__.detach().view(-1,batch_size,self.prd_width).mean(dim=0)
    #         zl_ = self.merge_head( e__ )
    #         z_ = torch.stack([zg_, zl_], dim=1).mean(dim=1)
    #     z.extend([z_,z_])
    #     for i in range(self.ens_size-2):
    #         z.append( zg_ )

    #     return f_.detach(), p, z, g_.detach()

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
        loss_tot_l = loss_l.detach()
        self.manual_backward( loss_l )

        self.log("f_", std_of_l2_normalized(f_))
        self.log("g_", std_of_l2_normalized(g_))
        self.log("z_", std_of_l2_normalized(z_))
        self.log("p_", std_of_l2_normalized(p_))
        self.log("||W||", self.rand_proj_q.weight.abs().mean())
        #self.log("||V||", self.merge_head.weight.abs().mean())

        if self.trainer.is_last_batch:
            opt.step()
            opt.zero_grad()
            sch.step() 
        elif (batch_idx + 1) % accumulate_grad_batches == 0:
            # momentum = cosine_schedule(self.current_epoch, max_epochs, 0.9, 1)
            # _do_momentum_update(self.rand_proj_n.weight, self.rand_proj_q.weight, momentum)
            opt.step()
            opt.zero_grad()
        
        self.log("pred_l", loss_tot_l,   prog_bar=True)

    def configure_optimizers(self):
        optim = torch.optim.SGD([
                {'params': self.backbone.parameters()},
                {'params': self.projection_head.parameters()},
                {'params': self.prediction_head.parameters()},
            ],            
            lr=0.2*lr_factor,
            momentum=0.9,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim] , [scheduler]
        

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
        pl.seed_everything(seed)
        dataset_train_ssl = create_dataset_train_ssl(BenchmarkModel)
        dataloader_train_ssl, dataloader_train_kNN, dataloader_test = get_data_loaders(
            batch_size=batch_size, dataset_train_ssl=dataset_train_ssl
        )
        benchmark_model = BenchmarkModel(dataloader_train_kNN, classes)

        # Save logs to: {CWD}/benchmark_logs/imagenet/{experiment_version}/{model_name}/
        # If multiple runs are specified a subdirectory for each run is created.
        sub_dir = model_name if n_runs <= 1 else f"{model_name}/run{seed}"
        logger = TensorBoardLogger(
            save_dir=os.path.join(logs_root_dir, "tinyimage"),
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
