import copy
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Identity
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import ConstantLR
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
                 running_stats:float=0.0,
                 ema_v2:bool=False,
                 momentum_head:bool=False,
                 identity_head:bool=False,
                 no_projection_head:bool=False,
                 n0:float = 1.00, n1:float = 1.00,
                 m0:float = 0.50, m1:float = 0.95,
                 prd_width:int = 256,
                 no_L2:bool=False,
                 no_ReLU_buttress:bool=False,
                 no_prediction_head:bool=False) -> None:
        super().__init__()
        self.save_hyperparameters('batch_size_per_device',
                                  'num_classes',
                                  'backbone',
                                  'n_local_views',
                                  'lr',
                                  'decay',
                                  'running_stats',
                                  'ema_v2',
                                  'momentum_head',
                                  'identity_head',
                                  'no_projection_head',
                                  'n0',
                                  'n1',
                                  'm0',
                                  'm1',
                                  'prd_width',
                                  'no_L2',
                                  'no_ReLU_buttress')
        self.lr = lr
        self.decay = decay
        self.batch_size_per_device = batch_size_per_device
        self.running_stats = running_stats
        self.ema_v2 = ema_v2
        if identity_head and momentum_head:
            raise Exception("Invalid Arguments, can't select identity and momentum")
        self.momentum_head = momentum_head                
        self.n0 = n0
        self.n1 = n1
        self.m0 = m0
        self.m1 = m1

        resnet, emb_width = backbones(backbone)
        self.emb_width  = emb_width # Used by eval classes
        
        self.ens_size = 2 + n_local_views
        self.backbone = resnet

        if no_projection_head:
            upd_width = self.emb_width
        else:
            upd_width = self.emb_width*4
        
        self.prd_width = prd_width

        if no_projection_head:
            self.projection_head = nn.Sequential()
        elif no_L2:
            self.projection_head = nn.Sequential(                    
                nn.Linear(emb_width, upd_width, False),
                nn.BatchNorm1d(upd_width),
                nn.ReLU(),
                nn.Linear(upd_width, upd_width),
            )
        else:
            self.projection_head = nn.Sequential(
                L2NormalizationLayer(),
                nn.Linear(emb_width, upd_width, False),
                nn.BatchNorm1d(upd_width),
                nn.ReLU(),
                nn.Linear(upd_width, upd_width),
            )
        
        #Use Batchnorm none-affine for centering
        if no_ReLU_buttress:
            self.buttress =  nn.Sequential(
                                nn.BatchNorm1d(upd_width, 
                                affine=False, 
                                momentum=self.running_stats, 
                                track_running_stats=(self.running_stats>0)),
                        )
        else:
            self.buttress =  nn.Sequential(
                                nn.BatchNorm1d(upd_width, 
                                affine=False, 
                                momentum=self.running_stats, 
                                track_running_stats=(self.running_stats>0)),
                                nn.LeakyReLU()
                        )
        if no_prediction_head:
            self.prediction_head = nn.Identity()
        else:
            self.prediction_head = nn.Linear(upd_width, self.prd_width, False)
        if identity_head:            
            if upd_width == prd_width:
                self.merge_head = nn.Linear(self.prd_width, self.prd_width)
                nn.init.eye_( self.merge_head.weight )
            elif upd_width > prd_width:
                #Identity matrix hack for if requires dimensionality reduction
                self.merge_head = nn.Sequential(
                    nn.AdaptiveAvgPool1d(self.prd_width),
                    nn.Linear(self.prd_width, self.prd_width),
                )
                nn.init.eye_( self.merge_head[1].weight )
            else:
                raise Exception("Invalid Arguments, can't select prd width larger than upd width")
        else:
            self.merge_head = nn.Linear(upd_width, self.prd_width)
            self.merge_head.weight.data = self.prediction_head.weight.data.clone()
        
        self.criterion = NegativeCosineSimilarity()

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
            if self.running_stats > 0.0:
                # Filthy hack to abuse the Batchnorm running stats
                self.buttress[0].training = False
                g0 = self.buttress( b[0] )
                g1 = self.buttress( b[1] )
                self.buttress[0].training = True
            else:
                g0 = g[0]
                g1 = g[1]
            zg0_ = self.merge_head( g0 )
            zg1_ = self.merge_head( g1 )
            if self.ema_v2:
                zg_ = 0.5*(zg0_+zg1_)
                if self.first_epoch:
                    self.embedding.weight[idx] = zg_.detach()
                else:
                    #For EMA 2.0
                    n = cosine_schedule(self.global_step, 
                                        self.trainer.estimated_stepping_batches, 
                                        self.n0, self.n1)
                    m = cosine_schedule(self.global_step, 
                                        self.trainer.estimated_stepping_batches, 
                                        self.m0, self.m1)
                    ze_ = self.embedding.weight[idx].clone()
                    if n < 1.0:
                        self.embedding.weight[idx] = (n)*zg_ + (1.-n)*ze_
                    else:
                        self.embedding.weight[idx] = zg_
                    #1 means only previous, 0 means only current
                    zg0_ = (m)*zg0_ + (1.-m)*ze_
                    zg1_ = (m)*zg1_ + (1.-m)*ze_
            z = [zg1_, zg0_]
            if self.ens_size > 2:
                zg_ = 0.5*(zg0_+zg1_)
                z.extend([zg_ for _ in range(self.ens_size-2)])
        return f0_, p, z

    def on_save_checkpoint(self, checkpoint):
        if self.ema_v2:
            del checkpoint['state_dict']['embedding.weight']

    def on_train_epoch_end(self):
        if self.ema_v2:
            self.first_epoch = False
        return super().on_train_epoch_end()

    def on_train_start(self):                
        if self.ema_v2:
            self.first_epoch = True
            self.embedding = nn.Embedding(len(self.trainer.train_dataloader.dataset), 
                                        self.prd_width, 
                                        dtype=torch.float16,
                                        device=self.device)        
        return super().on_train_start()

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
                    "lr": 0.1
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
                # end_value=0.1*self.lr
            ),
            "interval": "step",
        }
        # scheduler = {"scheduler": ConstantLR(optimizer=optimizer,factor=1.0,total_iters=1),"interval": "step",}
                     
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
                           global_crop_scale=(0.2, 1.0),
                           n_local_views=0,
                           cj_prob=0.0,
                           random_gray_scale=0.0,
                           solarization_prob=0.0,
                           gaussian_blur=(0.0, 0.0, 0.0)),
"Tiny-64-S": DINOTransform(global_crop_size=64,
                           n_local_views=0,
                           global_crop_scale=(0.2, 1.0),
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




