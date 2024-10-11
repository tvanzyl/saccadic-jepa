# -*- coding: utf-8 -*-
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

from lightly.data.collate import IJEPAMaskCollator
from lightly.models.modules.ijepa import IJEPABackbone, IJEPAPredictor
from lightly.transforms.ijepa_transform import IJEPATransform

from action_transform import ActionTransform, SimSimPTransform, RankingTransform

logs_root_dir = os.path.join(os.getcwd(), "benchmark_logs")

# Trade-off precision for performance.
# torch.set_float32_matmul_precision('medium')

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
batch_size = 511
lr_factor = batch_size / 128  # scales the learning rate linearly with batch size

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

# Use SimSiam augmentations
simsimp_transform = SimSimPTransform( 
    ens_size=5,
    input_size=32,
    gaussian_blur=0.0,
)

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

ijepa_transform = IJEPATransform(32)
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
        BarlowTwinsModel: byol_transform,
        BYOLModel: byol_transform,
        DCL: simclr_transform,
        DCLW: simclr_transform,
        DINOModel: dino_transform,
        FastSiamModel: fast_siam_transform,
        MocoModel: simclr_transform,
        NNCLRModel: simclr_transform,
        SimCLRModel: simclr_transform,
        SimSiamModel: simsiam_transform,
        SimSimPModel: simsimp_transform,
        SwaVModel: swav_transform,
        SMoGModel: smog_transform,
        IJEPA: ijepa_transform,
        SimMIM: simmim_transform,

    }
    transform = model_to_transform[model]
    return LightlyDataset(input_dir=path_to_train, transform=transform)

ijepa_collator = IJEPAMaskCollator(
    input_size=(32, 32),
    patch_size=4,
)

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

class SimMIM(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)

        # vit = torchvision.models.vit_b_32(pretrained=False)
        #https://github.com/facebookresearch/vissl/blob/main/configs/config/pretrain/dino/dino_16gpus_deits16.yaml
        vit = torchvision.models.vision_transformer._vision_transformer(
            patch_size=4,
            num_layers=12,
            num_heads=12,
            hidden_dim=132,
            mlp_dim=384,
            weights=None,
            progress=False,
            image_size=32,
        )

        decoder_dim = vit.hidden_dim
        self.mask_ratio = 0.75
        self.patch_size = vit.patch_size
        self.sequence_length = vit.seq_length
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        self.backbone = MaskedVisionTransformerTorchvision(vit=vit)

        # the decoder is a simple linear layer
        self.decoder = nn.Linear(vit.hidden_dim, vit.patch_size**2 * 3)

        # L1 loss as paper suggestion
        self.criterion = nn.L1Loss()

    def forward_encoder(self, images, batch_size, idx_mask):
        # pass all the tokens to the encoder, both masked and non masked ones
        return self.backbone.encode(images=images, idx_mask=idx_mask)

    def forward_decoder(self, x_encoded):
        return self.decoder(x_encoded)

    def training_step(self, batch, batch_idx):
        views = batch[0]
        images = views[0]  # views contains only a single view
        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )

        # Encoding...
        x_encoded = self.forward_encoder(images, batch_size, idx_mask)
        x_encoded_masked = utils.get_at_index(x_encoded, idx_mask)

        # Decoding...
        x_out = self.forward_decoder(x_encoded_masked)

        # get image patches for masked tokens
        patches = utils.patchify(images, self.patch_size)

        # must adjust idx_mask for missing class token
        target = utils.get_at_index(patches, idx_mask - 1)

        loss = self.criterion(x_out, target)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=1.5e-4)
        return optim


class MocoModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)

        # create a ResNet backbone and remove the classification head
        num_splits = 0 if sync_batchnorm else 8
        resnet = ResNetGenerator("resnet-18", num_splits=num_splits)
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1], nn.AdaptiveAvgPool2d(1)
        )

        # create a moco model based on ResNet
        self.projection_head = heads.MoCoProjectionHead(512, 512, 128)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        utils.deactivate_requires_grad(self.backbone_momentum)
        utils.deactivate_requires_grad(self.projection_head_momentum)

        # create our loss with the optional memory bank
        self.criterion = NTXentLoss(
            temperature=0.1,
            memory_bank_size=4096,
        )

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        return self.projection_head(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch

        # update momentum
        utils.update_momentum(self.backbone, self.backbone_momentum, 0.99)
        utils.update_momentum(self.projection_head, self.projection_head_momentum, 0.99)

        def step(x0_, x1_):
            x1_, shuffle = utils.batch_shuffle(x1_, distributed=distributed)
            x0_ = self.backbone(x0_).flatten(start_dim=1)
            x0_ = self.projection_head(x0_)

            x1_ = self.backbone_momentum(x1_).flatten(start_dim=1)
            x1_ = self.projection_head_momentum(x1_)
            x1_ = utils.batch_unshuffle(x1_, shuffle, distributed=distributed)
            return x0_, x1_

        # We use a symmetric loss (model trains faster at little compute overhead)
        # https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
        loss_1 = self.criterion(*step(x0, x1))
        loss_2 = self.criterion(*step(x1, x0))

        loss = 0.5 * (loss_1 + loss_2)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        params = list(self.backbone.parameters()) + list(
            self.projection_head.parameters()
        )
        optim = torch.optim.SGD(
            params,
            lr=6e-2 * lr_factor,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]


class SimCLRModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = ResNetGenerator("resnet-18")
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1], nn.AdaptiveAvgPool2d(1)
        )
        self.projection_head = heads.SimCLRProjectionHead(512, 512, 128)
        self.criterion = NTXentLoss()

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=6e-2 * lr_factor, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]


class L2NormalizationLayer(nn.Module):
    def __init__(self, dim=1, eps=1e-12):
        super(L2NormalizationLayer, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim, eps=self.eps)


class Twins(nn.Module):
    def __init__(self, emb_width, twintype, module):        
        super().__init__()        
        self.twintype = twintype        
        self.module = module
        self.ens_size = len(self.module)
        self.merge = heads.ProjectionHead(
                        [                
                            (emb_width*self.ens_size, emb_width, None, None),                            
                        ])

    def forward(self, x):
        embeddings = [m(x).flatten(start_dim=1) for m in self.module]
        if self.twintype == 'cat':
            out = torch.concat(embeddings, dim=1)
        elif self.twintype == 'rand':
            out = self.merge(torch.concat(embeddings, dim=1))
        elif self.twintype == 'first':
            out = embeddings[0]
        elif self.twintype == 'avg':
            out = embeddings[0] / self.ens_size
            for i in range(1, self.ens_size):
                out += embeddings[i] / self.ens_size
        else:
            raise Exception("Twin Type Not Supported")
        return out

class SimSimPModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        # self.automatic_optimization = False
        # create a ResNet backbone and remove the classification head
        emb_width = 512
        deb_width = 2048
        self.ens_size = 5
        self.scale_up = 5
        self.samebone = True
        self.mergetype = 'first'

        headbone = []
        for i in range(1 if self.samebone else self.ens_size):
            resnet = ResNetGenerator("resnet-18", width=emb_width/512.0)
            headbone.append(
                nn.Sequential(
                    *list(resnet.children())[:-1],
                    nn.AdaptiveAvgPool2d(1),
                )
            )
        if len(headbone) < self.ens_size:
            headbone = headbone*self.ens_size
        self.headbone = nn.ModuleList(headbone)
        self.backbone = Twins(emb_width, self.mergetype, self.headbone)
        
        projection_head = []
        for i in range(self.ens_size):
            projection_head.append(
                # nn.Identity()
                heads.ProjectionHead(
                    [
                        (emb_width, deb_width*self.scale_up, nn.BatchNorm1d(deb_width*self.scale_up), nn.ReLU(inplace=True)),
                        (deb_width*self.scale_up, emb_width, None, None),
                    ])
            )
        if len(projection_head) < self.ens_size:
            projection_head = projection_head*self.ens_size
        self.projection_head = nn.ModuleList(projection_head)

        prediction_head = []
        for i in range(self.ens_size):
            prediction_head.append(
                nn.Sequential(
                    nn.Linear(emb_width, deb_width, False), nn.BatchNorm1d(2048), nn.ReLU(inplace=True),
                    nn.Linear(deb_width, deb_width, False), 
                )
            )
        self.prediction_head = nn.ModuleList(prediction_head)

        merge_head = []
        merge_head_train = []
        for i in range(self.ens_size):
            bn1 = nn.BatchNorm1d(deb_width*self.scale_up)
            # bn2 = nn.BatchNorm1d(2048)
            merge_head.append(
                nn.Sequential(
                    nn.Linear(emb_width*(self.ens_size-1), deb_width*self.scale_up), bn1, nn.ReLU(inplace=True),
                    nn.Linear(deb_width*self.scale_up,                   deb_width), 
                )
            )
            merge_head_train.append(nn.ModuleList([bn1]))
        self.merge_head = nn.ModuleList(merge_head)
        self.merge_head_train = nn.ModuleList(merge_head_train)

        self.criterion = NegativeCosineSimilarity()

    def forward(self, x):
        g = self.forwarg_( x )
        p = self.forwarp_( g )
        z = self.forwarz_( g )
        return p, z, g
    
    def forwarg_(self, x):
        g = []
        for i in range(self.ens_size):
            f_ = self.headbone[i]( x[i] ).flatten(start_dim=1)
            g_ = self.projection_head[i]( f_ )
            g.append( g_ )
        return g

    def forwarz_(self, g):
        z = []        
        for i in range(self.ens_size):
            e_ = torch.concat([g[j].detach() for j in range(self.ens_size) if j != i], dim=1)
            z_ = self.merge_head[i]( e_ )
            z.append( z_ )
        return z

    def forwarp_(self, g):
        p = []        
        for i in range(self.ens_size):
            p_ = self.prediction_head[i]( g[i] )
            p.append( p_ )        
        return p

    def training_step(self, batch, batch_idx):
        x, _, _ = batch        
        # ((x), (x0,at0), (x1,at1), (x2,at2), (x3,at3)), _, _ = batch
        # opt = self.optimizers()
        # sch = self.lr_schedulers()

        with torch.no_grad():
            f = self.backbone(x[0]).flatten(start_dim=1)
        
        p, z, g = self.forward( x[1:self.ens_size+1] )

        loss_tot_l = 0
        scale = 0
        
        # opt.zero_grad()
        for i in range(0, len(p)):
            # opt[i].zero_grad()
            loss_l = self.criterion( p[i], z[i].detach() ) #increase diversity with abs()            
            loss_tot_l += loss_l
            scale += 1
            # self.manual_backward(loss_l, retain_graph=True)    
            # opt[i].step()
            # sch[i].step()

        loss_tot_l /= scale

        self.log("pred_l", loss_tot_l,   prog_bar=True)
        
        self.log("f_nm", f.norm(dim=1).mean())
        self.log("f_n", f.norm(dim=1).median())
        self.log("g_nm", g[0].norm(dim=1).mean())
        self.log("g_n", g[0].norm(dim=1).median())
        self.log("p_nm", p[0].norm(dim=1).mean())
        self.log("p_n", p[0].norm(dim=1).median())
        self.log("z_nm", z[0].norm(dim=1).median())
        self.log("z_n", z[0].norm(dim=1).median())
        
        self.log("f_s", f.std(dim=0).median())
        self.log("g_s", g[0].std(dim=0).median())
        self.log("p_s", p[0].std(dim=0).median())
        self.log("z_s", z[0].std(dim=0).median())
        
        self.log("g_d", self.criterion(g[0], g[1]))
        self.log("p_d", self.criterion(p[0], p[1]))
        self.log("z_d", self.criterion(z[0], z[1]))

        # self.log("emb_med", f.median(), prog_bar=True)
        # self.log("emb_mu",  f.mean(),   prog_bar=True)
        # self.log("emb_std", f.std(),    prog_bar=True)
        # self.log("emb_min", f.min(),    prog_bar=True)
        # self.log("emb_max", f.max(),    prog_bar=True)        
        return loss_tot_l

    def configure_optimizers(self):
        optims = []
        if self.samebone:
            optims.append({'params': self.headbone[0].parameters()})
        for i in range(self.ens_size):
            if not self.samebone:
                optims.append({'params': self.headbone[i].parameters()})
            optims.extend(
                [                
                {'params': self.projection_head[i].parameters()},
                {'params': self.prediction_head[i].parameters()}, #, 'weight_decay':5e-4},                
                {'params': self.merge_head_train[i].parameters()},                
                ])        
        optim = torch.optim.SGD(    
            optims,
            lr=6e-2*lr_factor,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)  
        return [optim], [scheduler]

    # def configure_optimizers(self):
    #     optim = []
    #     scheduler = []
    #     for i in range(self.ens_size):
    #         optim.append( torch.optim.SGD(
    #             [
    #             {'params': self.headbone[i].parameters()},
    #             {'params': self.projection_head[i].parameters()},
    #             {'params': self.prediction_head[i].parameters(), 'weight_decay':5e-4},
    #             {'params': self.merge_head[i].parameters()}
    #             ],                
    #             lr=6e-2*lr_factor,
    #             momentum=0.9,
    #             # weight_decay=5e-4,
    #         ) )
    #         scheduler.append(torch.optim.lr_scheduler.CosineAnnealingLR(optim[i], max_epochs))
    #         # scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, 6e-2*lr_factor, total_steps=19399)
    #     return optim, scheduler

class SimSiamModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = ResNetGenerator("resnet-18")
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1], nn.AdaptiveAvgPool2d(1)
        )
        self.prediction_head = heads.SimSiamPredictionHead(2048, 512, 2048)
        # use a 2-layer projection head for cifar10 as described in the paper
        self.projection_head = heads.ProjectionHead(
            [
                (512, 2048, nn.BatchNorm1d(2048), nn.ReLU(inplace=True)),
                (2048,2048, nn.BatchNorm1d(2048), None),
            ]
        )
        self.criterion = NegativeCosineSimilarity()

    def forward(self, x):
        f = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(f)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0, p0 = self.forward(x0)
        z1, p1 = self.forward(x1)
        loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(),
            lr=6e-2,  # no lr-scaling, results in better training stability
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]


class FastSiamModel(SimSiamModel):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)

    def training_step(self, batch, batch_idx):
        views, _, _ = batch
        features = [self.forward(view) for view in views]
        zs = torch.stack([z for z, _ in features])
        ps = torch.stack([p for _, p in features])

        loss = 0.0
        for i in range(len(views)):
            mask = torch.arange(len(views), device=self.device) != i
            loss += self.criterion(ps[i], torch.mean(zs[mask], dim=0)) / len(views)

        self.log("train_loss_ssl", loss)
        return loss


class BarlowTwinsModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = ResNetGenerator("resnet-18")
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1], nn.AdaptiveAvgPool2d(1)
        )
        # use a 2-layer projection head for cifar10 as described in the paper
        self.projection_head = heads.ProjectionHead(
            [
                (512, 2048, nn.BatchNorm1d(2048), nn.ReLU(inplace=True)),
                (2048, 2048, None, None),
            ]
        )

        self.criterion = BarlowTwinsLoss(gather_distributed=gather_distributed)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=6e-2 * lr_factor, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]

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
        
class BYOLModel(BenchmarkModuleP):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = ResNetGenerator("resnet-18")
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1], nn.AdaptiveAvgPool2d(1)
        )

        # create a byol model based on ResNet
        self.projection_head = heads.BYOLProjectionHead(512, 1024, 256)
        self.prediction_head = heads.BYOLPredictionHead(256, 1024, 256)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        utils.deactivate_requires_grad(self.backbone_momentum)
        utils.deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = NegativeCosineSimilarity()

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p

    def forward_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z

    def training_step(self, batch, batch_idx):
        utils.update_momentum(self.backbone, self.backbone_momentum, m=0.99)
        utils.update_momentum(
            self.projection_head, self.projection_head_momentum, m=0.99
        )
        (x0, x1), _, _ = batch
        p0 = self.forward(x0)
        z0 = self.forward_momentum(x0)
        p1 = self.forward(x1)
        z1 = self.forward_momentum(x1)
        loss = 0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        params = (
            list(self.backbone.parameters())
            + list(self.projection_head.parameters())
            + list(self.prediction_head.parameters())
        )
        optim = torch.optim.SGD(
            params,
            lr=6e-2 * lr_factor,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]


class SwaVModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = ResNetGenerator("resnet-18")
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1], nn.AdaptiveAvgPool2d(1)
        )

        self.projection_head = heads.SwaVProjectionHead(512, 512, 128)
        self.prototypes = heads.SwaVPrototypes(128, 512)  # use 512 prototypes

        self.criterion = SwaVLoss(sinkhorn_gather_distributed=gather_distributed)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.projection_head(x)
        x = nn.functional.normalize(x, dim=1, p=2)
        return self.prototypes(x)

    def training_step(self, batch, batch_idx):
        # normalize the prototypes so they are on the unit sphere
        self.prototypes.normalize()

        # the multi-crop dataloader returns a list of image crops where the
        # first two items are the high resolution crops and the rest are low
        # resolution crops
        multi_crops, _, _ = batch
        multi_crop_features = [self.forward(x) for x in multi_crops]

        # split list of crop features into high and low resolution
        high_resolution_features = multi_crop_features[:2]
        low_resolution_features = multi_crop_features[2:]

        # calculate the SwaV loss
        loss = self.criterion(high_resolution_features, low_resolution_features)

        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(),
            lr=1e-3 * lr_factor,
            weight_decay=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]


class NNCLRModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = ResNetGenerator("resnet-18")
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1], nn.AdaptiveAvgPool2d(1)
        )
        self.prediction_head = heads.NNCLRPredictionHead(256, 4096, 256)
        # use only a 2-layer projection head for cifar10
        self.projection_head = heads.ProjectionHead(
            [
                (512, 2048, nn.BatchNorm1d(2048), nn.ReLU(inplace=True)),
                (2048, 256, nn.BatchNorm1d(256), None),
            ]
        )

        self.criterion = NTXentLoss()
        self.memory_bank = modules.NNMemoryBankModule(size=4096)

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0, p0 = self.forward(x0)
        z1, p1 = self.forward(x1)
        z0 = self.memory_bank(z0, update=False)
        z1 = self.memory_bank(z1, update=True)
        loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(),
            lr=6e-2 * lr_factor,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]


class DINOModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = ResNetGenerator("resnet-18")
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1], nn.AdaptiveAvgPool2d(1)
        )
        self.head = self._build_projection_head()
        self.teacher_backbone = copy.deepcopy(self.backbone)
        self.teacher_head = self._build_projection_head()

        utils.deactivate_requires_grad(self.teacher_backbone)
        utils.deactivate_requires_grad(self.teacher_head)

        self.criterion = DINOLoss(output_dim=2048)

    def _build_projection_head(self):
        head = heads.DINOProjectionHead(512, 2048, 256, 2048, batch_norm=True)
        # use only 2 layers for cifar10
        head.layers = heads.ProjectionHead(
            [
                (512, 2048, nn.BatchNorm1d(2048), nn.GELU()),
                (2048, 256, None, None),
            ]
        ).layers
        return head

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

    def training_step(self, batch, batch_idx):
        utils.update_momentum(self.backbone, self.teacher_backbone, m=0.99)
        utils.update_momentum(self.head, self.teacher_head, m=0.99)
        views, _, _ = batch
        views = [view.to(self.device) for view in views]
        global_views = views[:2]
        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        param = list(self.backbone.parameters()) + list(self.head.parameters())
        optim = torch.optim.SGD(
            param,
            lr=6e-2 * lr_factor,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]


class DCL(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = ResNetGenerator("resnet-18")
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1], nn.AdaptiveAvgPool2d(1)
        )
        self.projection_head = heads.SimCLRProjectionHead(512, 512, 128)
        self.criterion = DCLLoss()

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=6e-2 * lr_factor, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]


class DCLW(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = ResNetGenerator("resnet-18")
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1], nn.AdaptiveAvgPool2d(1)
        )
        self.projection_head = heads.SimCLRProjectionHead(512, 512, 128)
        self.criterion = DCLWLoss()

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=6e-2 * lr_factor, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]


from sklearn.cluster import KMeans


class SMoGModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)

        # create a ResNet backbone and remove the classification head
        resnet = ResNetGenerator("resnet-18")
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1], nn.AdaptiveAvgPool2d(1)
        )

        # create a model based on ResNet
        self.projection_head = heads.SMoGProjectionHead(512, 2048, 128)
        self.prediction_head = heads.SMoGPredictionHead(128, 2048, 128)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        utils.deactivate_requires_grad(self.backbone_momentum)
        utils.deactivate_requires_grad(self.projection_head_momentum)

        # smog
        self.n_groups = 300
        memory_bank_size = 10000
        self.memory_bank = memory_bank.MemoryBankModule(size=(memory_bank_size, 128))
        # create our loss
        group_features = torch.nn.functional.normalize(
            torch.rand(self.n_groups, 128), dim=1
        )
        self.smog = heads.SMoGPrototypes(group_features=group_features, beta=0.99)
        self.criterion = nn.CrossEntropyLoss()

    def _cluster_features(self, features: torch.Tensor) -> torch.Tensor:
        features = features.cpu().numpy()
        kmeans = KMeans(self.n_groups).fit(features)
        clustered = torch.from_numpy(kmeans.cluster_centers_).float()
        clustered = torch.nn.functional.normalize(clustered, dim=1)
        return clustered

    def _reset_group_features(self):
        # see https://arxiv.org/pdf/2207.06167.pdf Table 7b)
        features = self.memory_bank.bank
        group_features = self._cluster_features(features.t())
        self.smog.set_group_features(group_features)

    def _reset_momentum_weights(self):
        # see https://arxiv.org/pdf/2207.06167.pdf Table 7b)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        utils.deactivate_requires_grad(self.backbone_momentum)
        utils.deactivate_requires_grad(self.projection_head_momentum)

    def training_step(self, batch, batch_idx):
        if self.global_step > 0 and self.global_step % 300 == 0:
            # reset group features and weights every 300 iterations
            self._reset_group_features()
            self._reset_momentum_weights()
        else:
            # update momentum
            utils.update_momentum(self.backbone, self.backbone_momentum, 0.99)
            utils.update_momentum(
                self.projection_head, self.projection_head_momentum, 0.99
            )

        (x0, x1), _, _ = batch

        if batch_idx % 2:
            # swap batches every second iteration
            x0, x1 = x1, x0

        x0_features = self.backbone(x0).flatten(start_dim=1)
        x0_encoded = self.projection_head(x0_features)
        x0_predicted = self.prediction_head(x0_encoded)
        x1_features = self.backbone_momentum(x1).flatten(start_dim=1)
        x1_encoded = self.projection_head_momentum(x1_features)

        # update group features and get group assignments
        assignments = self.smog.assign_groups(x1_encoded)
        group_features = self.smog.get_updated_group_features(x0_encoded)
        logits = self.smog(x0_predicted, group_features, temperature=0.1)
        self.smog.set_group_features(group_features)

        loss = self.criterion(logits, assignments)

        # use memory bank to periodically reset the group features with k-means
        self.memory_bank(x0_encoded, update=True)

        return loss

    def configure_optimizers(self):
        params = (
            list(self.backbone.parameters())
            + list(self.projection_head.parameters())
            + list(self.prediction_head.parameters())
        )
        optim = torch.optim.SGD(
            params,
            lr=0.01,
            momentum=0.9,
            weight_decay=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]

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


class IJEPA(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        
        ema = (0.996, 1.0)
        ipe_scale = 1.0
        ipe = len(dataloader_kNN)
        num_epochs = 10 
        momentum_scheduler = (
            ema[0] + i * (ema[1] - ema[0]) / (ipe * num_epochs * ipe_scale)
            for i in range(int(ipe * num_epochs * ipe_scale) + 1)
        )

        vit_predictor = torchvision.models.vision_transformer._vision_transformer(
            patch_size=4,
            num_layers=1,
            num_heads=12,
            hidden_dim=96,
            mlp_dim=128,
            weights=None,
            progress=False,
            image_size=32,
        )

        vit_encoder = torchvision.models.vision_transformer._vision_transformer(
            patch_size=4,
            num_layers=1,
            num_heads=12,
            hidden_dim=96,
            mlp_dim=128,
            weights=None,
            progress=False,
            image_size=32
        )

        # vit_predictor = torchvision.models.vit_b_32(pretrained=False)
        # vit_encoder = torchvision.models.vit_b_32(pretrained=False)

        self.encoder = IJEPABackbone.from_vit(vit_encoder)
        self.backbone = nn.Sequential(
            self.encoder, nn.Flatten()
        )

        self.predictor = IJEPAPredictor.from_vit_encoder(
            vit_predictor.encoder,
            (vit_predictor.image_size // vit_predictor.patch_size) ** 2,
        )
        self.target_encoder = copy.deepcopy(self.encoder)
        self.momentum_scheduler = momentum_scheduler

        self.criterion = nn.SmoothL1Loss()

    def forward_target(self, imgs, masks_enc, masks_pred):
        with torch.no_grad():
            h = self.target_encoder(imgs)
            h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
            B = len(h)
            # -- create targets (masked regions of h)
            h = apply_masks(h, masks_pred)
            h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
            return h

    def forward_context(self, imgs, masks_enc, masks_pred):
        z = self.encoder(imgs, masks_enc)
        z = self.predictor(z, masks_enc, masks_pred)
        return z

    def forward(self, x):
        imgs, masks_enc, masks_pred = x
        z = self.forward_context(imgs, masks_enc, masks_pred)
        h = self.forward_target(imgs, masks_enc, masks_pred)
        return z, h
    
    def training_step(self, batch, batch_idx):
        (imgs, _, _), masks_enc, masks_pred = batch
        self.update_target_encoder()
        z, h = self.forward((imgs, masks_enc, masks_pred))
        loss = self.criterion(z, h)
        self.log("train_loss_ssl", loss)
        return loss

    # def configure_optimizers(self):
    #     params = (
    #         list(self.encoder.parameters())
    #         + list(self.predictor.parameters())            
    #     )
    #     optim = torch.optim.AdamW(params, lr=1.5e-4)
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
    #     return [optim], [scheduler]

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=6e-2 * lr_factor, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]

    def update_target_encoder(self):
        with torch.no_grad():
            m = next(self.momentum_scheduler)
            for param_q, param_k in zip(
                self.encoder.parameters(), self.target_encoder.parameters()
            ):
                param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)

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
    # IJEPA,
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
        collator_ssl = ijepa_collator if BenchmarkModel is IJEPA else None        
        dataset_train_ssl = create_dataset_train_ssl(BenchmarkModel)
        dataloader_train_ssl, dataloader_train_kNN, dataloader_test = get_data_loaders(
            batch_size=batch_size, dataset_train_ssl=dataset_train_ssl, collator=collator_ssl
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
            # accumulate_grad_batches=128,
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


