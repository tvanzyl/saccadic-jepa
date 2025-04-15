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

from lightly.models import ResNetGenerator
from lightly.loss import NegativeCosineSimilarity
from lightly.models.utils import (
    get_weight_decay_parameters,
)

from lightly.transforms import DINOTransform
from lightly.utils.benchmarking import OnlineLinearClassifier
from lightly.utils.scheduler import CosineWarmupScheduler, cosine_schedule

class L2CenterNormLayer(nn.Module):
    def __init__(self, eps:float=1e-12):
        super(L2CenterNormLayer, self).__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:        
        c = x - x.mean(dim=0, keepdim=True)
        return c

class L2NormalizationLayer(nn.Module):
    def __init__(self, dim:int=1, eps:float=1e-12):
        super(L2NormalizationLayer, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:    
        return F.normalize(x, p=2, dim=self.dim, eps=self.eps)


def backbones(name):
    if name in ["resnetjie-9","resnetjie-18"]: 
        resnet = ResNetGenerator({"resnetjie-9":"resnet-9","resnetjie-18":"resnet-18"}[name])
        emb_width = resnet.linear.in_features        
        resnet = nn.Sequential(
            *list(resnet.children())[:-1], nn.AdaptiveAvgPool2d(1)
        )        
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
                 decay:float=1e-4) -> None:
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

        resnet, emb_width = backbones(backbone)
        self.emb_width  = emb_width # Used by eval classes

        
        prd_width ={1000:2048,
                    200:1024,
                    100:1024,
                    10:512,
                    }[num_classes]
        prd_width = 1024
        upd_width = prd_width*2
        self.ens_size = 2 + n_local_views

        self.backbone = resnet

        self.projection_head = nn.Sequential(
                nn.Linear(emb_width, emb_width*2, False),
                nn.BatchNorm1d(emb_width*2),
                nn.ReLU(),
                nn.Linear(emb_width*2, upd_width),
                L2NormalizationLayer(),
                nn.BatchNorm1d(upd_width, affine=False),
                nn.ReLU(),
            )                
        self.prediction_head = nn.Linear(upd_width, prd_width, False)
        self.merge_head = nn.Linear(upd_width, prd_width)
        # self.prediction_head.weight.data /= 3.0 #https://arxiv.org/pdf/2406.16468
        self.merge_head.weight.data = self.prediction_head.weight.data.clone()
        
        self.criterion = NegativeCosineSimilarity()

        self.online_classifier = OnlineLinearClassifier(feature_dim=emb_width, num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def forward_student(self, x: Tensor) -> Tensor:
        f = [self.backbone( x_ ).flatten(start_dim=1) for x_ in  x]
        f0_ = f[0].detach()
        g = [self.projection_head( f_ ) for f_ in f]
        p = [self.prediction_head( g_ ) for g_ in g]
        with torch.no_grad():
            zg0_ = self.merge_head( g[0] )
            zg1_ = self.merge_head( g[1] )
            z = [zg1_, zg0_]
            if self.ens_size>2:
                zg_ = 0.5*(zg0_+zg1_)
                for _ in range(self.ens_size-2):
                    z.append( zg_ )
        return f0_, p, z

    def training_step(
        self, batch: Tuple[List[Tensor], Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        x, targets, _ = batch
        
        f_, p, z = self.forward_student( x )
        
        loss = 0
        for xi in range(self.ens_size):
            p_ = p[xi]
            z_ = z[xi]
            loss += self.criterion( p_, z_ ) / self.ens_size
        
        self.log_dict(
            {"train_loss": loss},
            prog_bar=True,
            sync_dist=True,
            batch_size=len(targets),
        )

        # Online classification.
        cls_loss, cls_log = self.online_classifier.training_step(
            (f_, targets), batch_idx
        )        
        self.log_dict(cls_log, sync_dist=True, batch_size=len(targets))

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
        # Don't use weight decay for batch norm, bias parameters to improve performance.
        params, params_no_weight_decay = get_weight_decay_parameters(
            [self.backbone, self.prediction_head, ]#self.projection_head, ]
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
transform = DINOTransform(global_crop_scale=(0.2, 1), local_crop_scale=(0.05, 0.2))
transforms = {
"Cifar10": transform,
"Cifar100":transform,
"Tiny-64": DINOTransform(global_crop_size=128,
                         global_crop_scale=(0.2, 1.0),
                         local_crop_size=64,
                         local_crop_scale=(0.05, 0.2)),
"Tiny":    transform,
"Nette":   DINOTransform(global_crop_size=128,
                         global_crop_scale=(0.2, 1.0),
                         local_crop_size=64,
                         local_crop_scale=(0.05, 0.2)),
"Im100":   transform,
"Im1k":    transform,
"Im100-2": DINOTransform(global_crop_scale=(0.2, 1), 
                         n_local_views=0,
                         local_crop_scale=(0.05, 0.2)),
"Im1k-2":  DINOTransform(global_crop_scale=(0.2, 1), 
                         n_local_views=0,
                         local_crop_scale=(0.05, 0.2)),
}

val_identity  = T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]),
                ])

val_transform = T.Compose([
                    T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize(mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]),
                ])

val_transforms = {
"Cifar10": val_transform,
"Cifar100":val_transform,
"Tiny-64": T.Compose([
                T.Resize(128),T.ToTensor(),
                T.Normalize(mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]),
           ]),
"Tiny":    val_transform,
"Nette":   T.Compose([
                T.Resize(128),T.CenterCrop(128),T.ToTensor(),
                T.Normalize(mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]),
           ]),
"Im100":   val_transform,
"Im1k":    val_transform,
"Im100-2": val_transform,
"Im1k-2":  val_transform,
}

