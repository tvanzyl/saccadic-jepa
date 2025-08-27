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
                 warmup: int = 2,
                 backbone:str = "resnet-50",
                 n_local_views:int = 6,
                 lr:float = 0.15,
                 decay:float=1e-4,
                 running_stats:float=0.0,
                 ema_v2:bool=False,
                 momentum_head:bool=False,
                 identity_head:bool=False,
                 no_projection_head:bool=False,
                 alpha:float = 0.65,
                 n0:float = 1.00, n1:float = 1.00,                 
                 prd_width:int = 256,
                 prd_depth:int = 2,
                 upd_width:int = 2048,
                 L2:bool=False,
                 no_ReLU_buttress:bool=False,
                 no_prediction_head:bool=False,
                 JS:bool=False,
                 no_mem_bank:bool=False,
                 fwd_2:bool=False) -> None:
        super().__init__()
        self.save_hyperparameters('batch_size_per_device',
                                  'num_classes',
                                  'backbone',
                                  'n_local_views',
                                  'lr',
                                  'decay',
                                  'running_stats',
                                  'ema_v2', 'JS',
                                  'momentum_head',
                                  'identity_head',
                                  'no_projection_head',
                                  'no_prediction_head',
                                  'alpha',
                                  'n0', 'n1',                                  
                                  'prd_width', "prd_depth",
                                  "upd_width",
                                  'L2',
                                  'no_ReLU_buttress',
                                  'no_mem_bank',
                                  'fwd_2')
        self.warmup = warmup
        self.lr = lr
        self.decay = decay
        self.batch_size_per_device = batch_size_per_device
        self.running_stats = running_stats
        self.ema_v2 = ema_v2
        self.JS = JS
        self.mem_bank = not no_mem_bank        
        self.fwd_2 = fwd_2
        self.momentum_head = momentum_head                
        self.alpha = alpha
        self.n0 = n0
        self.n1 = n1        

        if identity_head and momentum_head:
            raise Exception("Invalid Arguments, can't select identity and momentum")
        if JS and ema_v2:
            raise Exception("Invalid Arguments, can't select JS and EMA")
        if JS and no_mem_bank and not fwd_2:
            raise Exception("Need One of Fwd 2 or Mem Bank")

        resnet, emb_width = backbones(backbone)
        self.emb_width  = emb_width # Used by eval classes
        
        self.ens_size = 2 + n_local_views
        self.backbone = resnet

        if no_projection_head:
            upd_width = self.emb_width
        
        self.prd_width = prd_width

        if no_projection_head:
            self.projection_head = nn.Sequential()
        else:
            if prd_depth == 2:
                projection_head = [nn.Linear(emb_width, upd_width, False),
                                   nn.BatchNorm1d(upd_width),
                                   nn.ReLU(),
                                   nn.Linear(upd_width, upd_width, False),
                                   nn.BatchNorm1d(upd_width),
                                   nn.ReLU(),
                                   nn.Linear(upd_width, upd_width),]
            elif prd_depth == 1:
                projection_head = [nn.Linear(emb_width, upd_width, False),
                                   nn.BatchNorm1d(upd_width),
                                   nn.ReLU(),
                                   nn.Linear(upd_width, upd_width),]
            else:
                raise Exception("Selected Prediction Depth Not Supported")
                
            if L2:
                projection_head.insert(0, L2NormalizationLayer())

            self.projection_head = nn.Sequential(          
                                    *projection_head
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
                                nn.ReLU(),                                
                        )
        if no_prediction_head:
            self.prediction_head = nn.AdaptiveAvgPool1d(self.prd_width)
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
        if self.fwd_2:
            y = x[2:]
            x = x[:2]
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
                g0 = g[0].detach()
                g1 = g[1].detach()                
            zg0_ = self.merge_head( g0 )
            zg1_ = self.merge_head( g1 )
             
            if self.fwd_2:
                fy = [self.backbone( y_ ).flatten(start_dim=1) for y_ in y]
                by = [self.projection_head( fy_ ) for fy_ in fy]
                gy = [self.buttress( by_ ) for by_ in by]
                zy = [self.merge_head( gy_ ) for gy_ in gy]
                ze2_ = (zy[0]+zy[1])/2.0
                    
            if self.JS:
                # For James-Stein                
                zg_ = 0.5*(zg0_+zg1_)
                if self.first_epoch:
                    self.embedding[idx] = zg_
                else:                    
                    if self.fwd_2 and self.mem_bank:
                        ze_ = (self.embedding[idx] + ze2_)/2.0
                    elif self.mem_bank:
                        ze_ = self.embedding[idx].clone()
                    elif self.fwd_2:
                        ze_ = ze2_

                    # EWM-A/V https://fanf2.user.srcf.net/hermes/doc/antiforgery/stats.pdf
                    zvr_ = self.embedding_var[idx]

                    zdf0_ = zg0_ - ze_
                    zic0_ = self.alpha * zdf0_
                    sigma0_ = torch.mean((1.0 - self.alpha) * (zvr_ + zdf0_ * zic0_), dim=1, keepdim=True)

                    zdf1_ = zg1_ - ze_
                    zic1_ = self.alpha * zdf1_
                    sigma1_ = torch.mean((1.0 - self.alpha) * (zvr_ + zdf1_ * zic1_), dim=1, keepdim=True)
                    
                    sigma_ = (sigma0_+sigma1_)/2.0
                    zic_ = (zic0_+zic1_)/2.0                    
                    self.embedding[idx] = (ze_ + zic_)
                    self.embedding_var[idx] = sigma_

                    # https://openaccess.thecvf.com/content/WACV2024/papers/Khoshsirat_Improving_Normalization_With_the_James-Stein_Estimator_WACV_2024_paper.pdf
                    norm0_ = torch.linalg.vector_norm(zg0_-ze_, dim=1, keepdim=True)**2
                    norm1_ = torch.linalg.vector_norm(zg1_-ze_, dim=1, keepdim=True)**2
                    
                    sigma_ = (self.prd_width-2.0)*sigma_                    
                    
                    n0 = torch.maximum(1.0 - sigma_/norm0_, torch.tensor(0.0))
                    n1 = torch.maximum(1.0 - sigma_/norm1_, torch.tensor(0.0))

                    self.log_dict({"JS_n0_n1":0.5*n0.mean() + 0.5*n1.mean()})

                    zg0_ = n0*zg0_ + (1.-n0)*ze_
                    zg1_ = n1*zg1_ + (1.-n1)*ze_
                    

            if self.ema_v2:
                n = cosine_schedule(self.global_step, 
                                    self.trainer.estimated_stepping_batches, 
                                    self.n0, self.n1)            
                zg_ = 0.5*(zg0_+zg1_)
                if self.first_epoch:
                    self.embedding[idx] = zg_.detach()
                else:
                    #For EMA 2.0
                    if self.fwd_2 and self.mem_bank:
                        ze_ = (self.embedding[idx] + ze2_)/2.0
                    elif self.mem_bank:
                        ze_ = self.embedding[idx]
                    elif self.fwd_2:
                        ze_ = ze2_

                    #1 means only previous, 0 means only current
                    zg0_ = (n)*zg0_ + (1.-n)*ze_
                    zg1_ = (n)*zg1_ + (1.-n)*ze_

                    if self.alpha < 1.0:
                        self.embedding[idx] = (self.alpha)*zg_ + (1.-self.alpha)*ze_
                    else:
                        self.embedding[idx] = zg_
            z = [zg1_, zg0_]
            if self.ens_size > 2 and not self.fwd_2:
                zg_ = 0.5*(zg0_+zg1_)
                z.extend([zg_ for _ in range(self.ens_size-2)])
            assert len(p)==len(z)
        return f0_, p, z

    def on_train_epoch_end(self):
        if self.ema_v2 or self.JS:
            self.first_epoch = False
        return super().on_train_epoch_end()

    def on_train_start(self):                
        if self.ema_v2 or self.JS:
            self.first_epoch = True            
            N = len(self.trainer.train_dataloader.dataset)
            self.embedding      = torch.empty((N, self.prd_width),
                                        dtype=torch.float16,
                                        device=self.device)
            self.embedding_var  = torch.zeros((N, 1), 
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
                    [self.backbone, self.prediction_head,] #  self.projection_head]
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
                    * self.warmup
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
def train_transform(size, scale=(0.2, 1.0)):
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
"Cifar10":  DINOTransform(global_crop_size=32,
                          global_crop_scale=(0.2, 1.0),
                          n_local_views=0,
                          gaussian_blur=(0.0, 0.0, 0.0),),
"Cifar100": DINOTransform(global_crop_size=32,
                          global_crop_scale=(0.2, 1.0),
                          n_local_views=0,
                          gaussian_blur=(0.0, 0.0, 0.0),),
"Tiny":     DINOTransform(global_crop_size=64,
                          global_crop_scale=(0.2, 1.0),
                          local_crop_size=32,
                          local_crop_scale=(0.05, 0.2),
                          gaussian_blur=(0.5, 0.0, 0.0),),
"Tiny-W":   DINOTransform(global_crop_size=64,
                          global_crop_scale=(0.2, 1.0),
                          n_local_views=0,
                          cj_prob=0.0,
                          random_gray_scale=0.0,
                          solarization_prob=0.0,
                          gaussian_blur=(0.0, 0.0, 0.0)),
"Tiny-S":   DINOTransform(global_crop_size=64,
                          n_local_views=0,
                          global_crop_scale=(0.2, 1.0),
                          gaussian_blur=(0.5, 0.0, 0.0),),
"Tiny-4":   DINOTransform(global_crop_size=64,
                          global_crop_scale=(0.2, 1.0),
                          local_crop_size=64,
                          local_crop_scale=(0.2, 1.0),
                          n_local_views=2,                          
                          gaussian_blur=(0.5, 0.0, 0.0),),
"STL":      DINOTransform(global_crop_size=96,
                          global_crop_scale=(0.2, 1.0),
                          local_crop_size=48,
                          local_crop_scale=(0.05, 0.2),
                          gaussian_blur=(0.5, 0.0, 0.0),),
"STL-2":    DINOTransform(global_crop_size=96,
                          n_local_views=0,
                          global_crop_scale=(0.2, 1.0),),
"Nette":    DINOTransform(global_crop_size=128,
                          global_crop_scale=(0.2, 1.0),
                          local_crop_size=64,
                          local_crop_scale=(0.05, 0.2),),
"Im100":    DINOTransform(global_crop_scale=(0.2, 1), 
                          local_crop_scale=(0.05, 0.2)),
"Im100-2":  DINOTransform(global_crop_scale=(0.20, 1.0),
                          n_local_views=0),
"Im100-4":  DINOTransform(global_crop_scale=(0.2, 1.0),
                          local_crop_scale=(0.2, 1.0),
                          n_local_views=2,),
"Im1k":     DINOTransform(global_crop_scale=(0.2, 1), 
                          local_crop_scale=(0.05, 0.2)),
"Im1k-2":   DINOTransform(global_crop_scale=(0.05, 1.0),
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
