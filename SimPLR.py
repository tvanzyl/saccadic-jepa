import copy
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Identity
from torch.optim import SGD, AdamW
from torch.optim.optimizer import Optimizer
from torchvision import transforms as T
from torchvision.models import resnet50, resnet34, resnet18

from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.models._momentum import _do_momentum_update
from lightly.models import ResNetGenerator
from lightly.loss import NegativeCosineSimilarity
from lightly.models.utils import (
    get_weight_decay_parameters,
)

from lightly.transforms import DINOTransform
from lightly.utils.benchmarking import OnlineLinearClassifier
from lightly.utils.scheduler import CosineWarmupScheduler, cosine_schedule

def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """
    def __getitem__(self, index):
        data, target, fname = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })

class L2NormalizationLayer(nn.Module):
    def __init__(self, dim:int=1, eps:float=1e-12):
        super(L2NormalizationLayer, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:    
        return F.normalize(x, p=2, dim=self.dim, eps=self.eps)


def backbones(name):
    if name in ["resnetjie-9","resnetjie-18"]: 
        # resnet = ResNetGenerator({"resnetjie-9":"resnet-9","resnetjie-18":"resnet-18"}[name])
        # emb_width = resnet.linear.in_features        
        # resnet = nn.Sequential(
        #     *list(resnet.children())[:-1], nn.AdaptiveAvgPool2d(1)
        # )        
        resnet = {"resnetjie-9" :resnet18, 
                  "resnetjie-18":resnet18}[name]()
        emb_width = resnet.fc.in_features
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        resnet.maxpool = nn.Sequential()
        resnet.fc = Identity()
    elif name in ["resnet-18", "resnet-34", "resnet-50"]: 
        resnet = {"resnet-18":resnet18, 
                  "resnet-34":resnet34, 
                  "resnet-50":resnet50}[name]()
        emb_width = resnet.fc.in_features
        resnet.fc = Identity()
    else:
        raise NotImplemented("Backbone Not Supported")

    return resnet, emb_width


class SimPLR(LightningModule):
    def __init__(self, batch_size_per_device: int, 
                 num_classes: int, 
                 backbone:str = "resnet-50",
                 n_local_views:int = 6,
                 lr:float = 0.15,
                 decay:float=1e-4,
                 running_stats:bool=False,
                 ema_v2:bool=False,
                 momentum_head:bool=False,) -> None:
        super().__init__()
        self.save_hyperparameters('batch_size_per_device',
                                  'num_classes',
                                  'backbone',
                                  'n_local_views',
                                  'lr',
                                  'decay')
        self.lr = lr
        self.decay = decay
        self.batch_size_per_device = batch_size_per_device
        self.running_stats = running_stats
        self.ema_v2 = ema_v2
        self.momentum_head = momentum_head

        resnet, emb_width = backbones(backbone)
        self.emb_width  = emb_width # Used by eval classes

        self.prd_width = 256
        upd_width = 2048
        self.ens_size = 2 + n_local_views

        self.backbone = resnet

        self.projection_head = nn.Sequential(
                nn.Linear(emb_width, upd_width, False),
                nn.BatchNorm1d(upd_width),
                nn.ReLU(),
                nn.Linear(upd_width, upd_width),
                L2NormalizationLayer(),
            )
        #Use Batchnorm none-affine for centering
        self.buttress =  nn.Sequential(                
                nn.BatchNorm1d(upd_width, affine=False),
                nn.LeakyReLU()
        )
        self.prediction_head = nn.Linear(upd_width, self.prd_width, False)        
        self.merge_head = nn.Linear(upd_width, self.prd_width)      
        self.merge_head.weight.data = self.prediction_head.weight.data.clone()
        #Uncomment this line for identity teacher
        # nn.init.eye_( self.merge_head.weight )
        
        self.criterion = NegativeCosineSimilarity()

        #Inc case we need to load model before the bug fix
        # self.embedding = nn.Embedding(100000, 
        #                               self.prd_width, 
        #                               dtype=torch.float16,
        #                               device=self.device)

        self.online_classifier = OnlineLinearClassifier(feature_dim=emb_width, num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def forward_student(self, x: Tensor, idx: Tensor) -> Tensor:
        f = [self.backbone( x_ ).flatten(start_dim=1) for x_ in  x]
        f0_ = f[0].detach()
        b = [self.projection_head( f_ ) for f_ in f]
        g = [self.buttress( b_ ) for b_ in b]
        p = [self.prediction_head( g_ ) for g_ in g]
        with torch.no_grad():
            if self.running_stats:
                # Filthy hack to abuse the Batchnorm running stats
                self.buttress[0].training = False
                g0 = self.buttress( b[0] )
                g1 = self.buttress( b[1] )
                self.buttress[0].training = True
                zg0_ = self.merge_head( g0 )
                zg1_ = self.merge_head( g1 )
            else:
                zg0_ = self.merge_head( g[0] )
                zg1_ = self.merge_head( g[1] )
            if self.ema_v2:
                zg_ = 0.5*(zg0_+zg1_)
                momentum = cosine_schedule(self.global_step, self.trainer.estimated_stepping_batches, 0.95, 1.0)
                # momentum = 0.95
                m = 1.-momentum #0 means only current, 1 means only previous
                ze_ = self.embedding.weight[idx].clone()
                zg0_ = (1.-m)*zg0_ + m*ze_
                zg1_ = (1.-m)*zg1_ + m*ze_
                self.embedding.weight[idx] = (1.-m)*(zg0_+zg1_) + m*ze_
                # _do_momentum_update(self.embedding.weight[idx], zg_, m) 
            z = [zg1_, zg0_]
            if self.ens_size>2:
                zg_ = 0.5*(zg0_+zg1_)
                for _ in range(self.ens_size-2):
                    z.append( zg_ )
        
        #Uncomment for EMA 2.0
        # p.extend([p[0], p[1]])
        # with torch.no_grad():
        #     ze_ = self.embedding.weight[idx].clone()
        #     z.extend([ze_, ze_])
        #     _do_momentum_update(self.embedding.weight[idx], 0.5*(zg0_+zg1_), 0.1)
            # 

        return f0_, p, z

    def on_save_checkpoint(self, checkpoint):
        if self.ema_v2:
            del checkpoint['state_dict']['embedding.weight']

    def on_train_start(self):
        if self.ema_v2:
            self.embedding = nn.Embedding(len(self.trainer.train_dataloader.dataset), 
                                        self.prd_width, 
                                        dtype=torch.float16,
                                        device=self.device)

    def training_step(
        self, batch: Tuple[List[Tensor], Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        x, targets, idx = batch
        
        f0_, p, z = self.forward_student( x, idx )
        
        loss = 0
        for xi in range(len(z)):
            p_ = p[xi]
            z_ = z[xi]
            loss += self.criterion( p_, z_ ) / len(z)
        
        self.log_dict(
            {"train_loss": loss},
            prog_bar=True,
            sync_dist=True,
            batch_size=len(targets),
        )

        # Online classification.
        cls_loss, cls_log = self.online_classifier.training_step(
            (f0_, targets), batch_idx
        )        
        self.log_dict(cls_log, sync_dist=True, batch_size=len(targets))

        #These lines give us classical EMA v1
        if self.momentum_head:
            momentum = cosine_schedule(self.global_step, self.trainer.estimated_stepping_batches, 0.996, 1)
            _do_momentum_update(self.merge_head.weight, self.prediction_head.weight, momentum)

        return loss + cls_loss

    def validation_step(
        self, batch: Tuple[Tensor, Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        images, targets = batch[0], batch[1]
        features = self.forward(images).flatten(start_dim=1)
        cls_loss, cls_log = self.online_classifier.validation_step(
            (features.detach(), targets), batch_idx
        )
        self.log_dict(cls_log, prog_bar=True, sync_dist=True, batch_size=len(targets))
        return cls_loss

    def configure_optimizers(self):
        params, params_no_weight_decay = get_weight_decay_parameters(
                    [self.backbone, self.prediction_head, ]# self.projection_head]
                )
        optimizer = SGD(        
            [
                {"name": "simplr", "params": params},
                {
                    "name": "proj", 
                    "params": self.projection_head.parameters(),
                },
                {
                    "name": "simplr_no_weight_decay",
                    "params": params_no_weight_decay,
                    "weight_decay": 0.0,
                },     
                {
                    "name": "online_classifier",
                    "params": self.online_classifier.parameters(),                    
                    "weight_decay": 0.0,
                    "lr":0.1
                },
            ],            
            lr=self.lr * self.batch_size_per_device * self.trainer.world_size / 256,
            momentum=0.9,
            weight_decay=self.decay,
        )         
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                # warmup_epochs=0,
                warmup_epochs=int(
                      self.trainer.estimated_stepping_batches
                    / self.trainer.max_epochs
                    * 10
                ),
                max_epochs=int(self.trainer.estimated_stepping_batches),
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]

# For ResNet50 we adjust crop scales as recommended by the authors:
# https://github.com/facebookresearch/dino#resnet-50-and-other-convnets-trainings
# transform = DINOTransform(global_crop_scale=(0.14, 1), local_crop_scale=(0.05, 0.14), n_local_views=n_local_views)
transform   = DINOTransform(global_crop_scale=(0.2, 1), local_crop_scale=(0.05, 0.2))
transform32 = DINOTransform(global_crop_size=32,
                    global_crop_scale=(0.2, 1.0),
                    n_local_views=0,
                    gaussian_blur=(0.0, 0.0, 0.0),
                )
transform64 = DINOTransform(global_crop_size=64,
                    global_crop_scale=(0.3, 1.0),
                    local_crop_size=32,
                    local_crop_scale=(0.08, 0.3),
                    gaussian_blur=(0.0, 0.0, 0.0),
                )
transform96 = DINOTransform(global_crop_size=96,
                    global_crop_scale=(0.3, 1.0),
                    local_crop_size=48,
                    local_crop_scale=(0.08, 0.3),
                ) 
transform128= DINOTransform(global_crop_size=128,
                    global_crop_scale=(0.2, 1.0),
                    local_crop_size=64,
                    local_crop_scale=(0.05, 0.2),                    
                )

def train_transform(size, scale=(0.08, 1.)):
    return T.Compose([
                    T.RandomResizedCrop(size, scale=scale),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]),
                ])
val_identity  = lambda size: T.Compose([
                    T.Resize(size), T.ToTensor(),
                    T.Normalize(mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]),
                ])
val_transform = T.Compose([
                    T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize(mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]),
                ])


transforms = {
"Cifar10":   transform32,
"Cifar100":  transform32,
"Tiny":      transform64,
"Tiny-64-W": DINOTransform(global_crop_size=64,
                           global_crop_scale=(0.08, 1.0),
                           n_local_views=0,
                           cj_prob=0.0,
                           random_gray_scale=0.0,
                           solarization_prob=0.0,
                           gaussian_blur=(0.0, 0.0, 0.0)),
"Tiny-64-S": DINOTransform(global_crop_size=64,
                           n_local_views=0,
                           global_crop_scale=(0.08, 1.0),
                           gaussian_blur=(0.0, 0.0, 0.0),),
"STL":       transform96,
"STL-S":     DINOTransform(global_crop_size=96,
                           n_local_views=0,
                           global_crop_scale=(0.08, 1.0),),
"Nette":     transform128,
"Im100":     transform,
"Im1k":      transform,
"Im100-2":   DINOTransform(global_crop_scale=(0.08, 1),
                           n_local_views=0),
"Im1k-2":    DINOTransform(global_crop_scale=(0.08, 1),
                           n_local_views=0),
}

val_transforms = {
"Cifar10":   val_identity(32),
"Cifar100":  val_identity(32),
"Tiny":      val_identity(64),
"STL":       val_identity(96),
"Nette":     val_identity(128),
"Im100":     val_transform,
"Im1k":      val_transform,
}




