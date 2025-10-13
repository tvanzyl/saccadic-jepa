import copy
import math
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
from lightly.loss.ntx_ent_loss import NTXentLoss
from lightly.loss.hypersphere_loss import HypersphereLoss
from lightly.models.utils import (
    get_weight_decay_parameters,
)

from lightly.transforms import DINOTransform
from lightly.utils.benchmarking import OnlineLinearClassifier
from lightly.utils.scheduler import CosineWarmupScheduler, cosine_schedule

from action_transform import JSREPATransform

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
        # resnet.fc = nn.BatchNorm1d(emb_width)
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
                 ema_v2:bool=False,
                 momentum_head:bool=False,
                 identity_head:bool=False,
                 no_projection_head:bool=False,
                 alpha:float = 0.65,
                 n0:float = 1.00, n1:float = 1.00,                 
                 prd_width:int = 256,
                 prj_depth:int = 2,
                 prj_width:int = 2048,
                 L2:bool=False,M2:bool=False,
                 no_ReLU_buttress:bool=False,
                 no_prediction_head:bool=False,
                 JS:bool=False, JS_INV:bool=False,
                 emm:bool=False, emm_v:int=0,
                 fwd:int=0,
                 asm:bool=False,
                 loss:str="negcosine",
                 nn_init:str="fan-in") -> None:
        super().__init__()
        self.save_hyperparameters('batch_size_per_device',
                                  'num_classes',
                                  'backbone',
                                  'n_local_views',
                                  'lr',
                                  'decay',                                  
                                  'ema_v2', 'JS',
                                  'momentum_head',
                                  'identity_head',
                                  'no_projection_head',
                                  'no_prediction_head',
                                  'alpha',
                                  'n0', 'n1',                                  
                                  'prd_width', 
                                  "prj_depth", "prj_width",
                                  'L2','M2',
                                  'no_ReLU_buttress',
                                  'emm', 'emm_v',
                                  'fwd',
                                  'asm', 
                                  'loss',
                                  'nn_init')
        self.warmup = warmup
        self.lr = lr
        self.decay = decay
        self.batch_size_per_device = batch_size_per_device        
        self.ema_v2 = ema_v2
        self.JS = JS
        self.JS_INV = JS_INV
        self.emm = emm
        self.emm_v = emm_v
        self.fwd = fwd
        self.asm = asm
        self.momentum_head = momentum_head                
        self.alpha = alpha
        self.n0 = n0
        self.n1 = n1        

        if identity_head and momentum_head:
            raise Exception("Invalid Arguments, can't select identity and momentum")
        if JS and ema_v2:
            raise Exception("Invalid Arguments, can't select JS and EMA")
        if JS and not emm and fwd == 0:
            raise Exception("Invalid Arguments, Need One of Fwd or EMM with JS")
        if self.asm and not self.emm:
            raise Exception("Invalid Arguments, Need EMM with Asm")
        if self.asm and self.fwd > 0:
            raise Exception("Invalid Arguments, can't have Asm with Fwd")

        resnet, emb_width = backbones(backbone)
        self.emb_width  = emb_width # Used by eval classes        
        
        self.backbone = resnet

        if no_projection_head:
            prj_width = self.emb_width
        
        self.prd_width = prd_width

        if no_projection_head:
            self.projection_head = nn.Sequential()
        else:
            if prj_depth == 2:
                projection_head = [nn.Linear(emb_width, prj_width, False),
                                   nn.BatchNorm1d(prj_width),
                                   nn.ReLU(),
                                   nn.Linear(prj_width, prj_width, False),
                                   nn.BatchNorm1d(prj_width),
                                   nn.ReLU(),
                                   nn.Linear(prj_width, prj_width),]
            elif prj_depth == 1:
                projection_head = [nn.Linear(emb_width, prj_width, False),
                                   nn.BatchNorm1d(prj_width),
                                   nn.ReLU(),
                                   nn.Linear(prj_width, prj_width),]
            elif prj_depth == 0:
                projection_head = [nn.Linear(emb_width, prj_width),]
            else:
                raise NotImplementedError("Selected Prediction Depth Not Supported")
                
            if L2:
                projection_head.insert(0, L2NormalizationLayer())
            if M2:
                # projection_head.insert(0, nn.BatchNorm1d(self.emb_width))
                projection_head.append(L2NormalizationLayer())

            self.projection_head = nn.Sequential(          
                                    *projection_head
                                )
        
        #Use Batchnorm none-affine for centering
        if no_ReLU_buttress:
            self.buttress =  nn.Sequential(
                                nn.BatchNorm1d(prj_width, 
                                affine=False),
                        )
        else:
            self.buttress =  nn.Sequential(
                                nn.BatchNorm1d(prj_width, 
                                affine=False),
                                nn.ReLU(),
                        )
        if no_prediction_head:
            self.prediction_head = nn.AdaptiveAvgPool1d(self.prd_width)
        else:
            self.prediction_head = nn.Linear(prj_width, self.prd_width, False)
        if identity_head:
            if prj_width == prd_width:
                self.merge_head = nn.Linear(self.prd_width, self.prd_width)
                nn.init.eye_( self.merge_head.weight )
            elif prj_width > prd_width:
                #Identity matrix hack for if requires dimensionality reduction
                self.merge_head = nn.Sequential(
                    nn.AdaptiveAvgPool1d(self.prd_width),
                    nn.Linear(self.prd_width, self.prd_width),
                )
                nn.init.eye_( self.merge_head[1].weight )
            else:
                raise NotImplementedError("Invalid Arguments, can't select prd width larger than prj width")
        else:
            self.merge_head = nn.Linear(prj_width, self.prd_width)
            if nn_init == "fan-in":
                bound = 1 / math.sqrt(self.prediction_head.weight.size(1))
                nn.init.uniform_(self.prediction_head.weight, -bound, bound)
            elif nn_init == "fan-out":
                bound = 1 / math.sqrt(self.prediction_head.weight.size(0))                
                nn.init.uniform_(self.prediction_head.weight, -bound, bound)
            elif nn_init == "gauss-out":
                bound = 1 / math.sqrt(self.prediction_head.weight.size(0))
                nn.init.normal_(self.prediction_head.weight, 0, bound)
            self.merge_head.weight.data = self.prediction_head.weight.data.clone()
        
        self.criterion = {"negcosine":NegativeCosineSimilarity(),   
                          "nxtent":NTXentLoss(memory_bank_size=0),
                          "hypersphere":HypersphereLoss()}[loss]

        self.online_classifier = OnlineLinearClassifier(feature_dim=emb_width, num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def forward_student(self, x: List[Tensor], idx: Tensor) -> Tensor:
        views = len(x)

        # Two globals
        f = [self.backbone( x_ ).flatten(start_dim=1) for x_ in x[:2]]
        f0_ = f[0].detach()        
        if views > self.fwd + 2: # MultiCrops
            f.extend([self.backbone( x_ ).flatten(start_dim=1) for x_ in x[self.fwd+2:]])
        b = [self.projection_head( f_ ) for f_ in f]
        g = [self.buttress( b_ ) for b_ in b]
        p = [self.prediction_head( g_ ) for g_ in g]

        # Fwds Only
        if self.fwd > 0:
            with torch.no_grad():
                f_fwd = [self.backbone( x_ ).flatten(start_dim=1) for x_ in x[2:self.fwd+2]]
                b_fwd = [self.projection_head( f_ ) for f_ in f_fwd]
                g_fwd = [self.buttress( b_ ) for b_ in b_fwd]
                z_fwd = [self.merge_head( g_ ) for g_ in g_fwd]

        with torch.no_grad(): 
            z = [self.merge_head( g_.detach() ) for g_ in g]
            zg0_ = z[0]
            zg1_ = z[1]

            if self.JS: # For James-Stein
                if self.first_epoch:
                    self.embedding[idx] = 0.5*(zg0_+zg1_)
                    # self.embedding_diff[idx] = 0.5*(zg0_+zg1_)
                else:
                    # EWM-A/V https://fanf2.user.srcf.net/hermes/doc/antiforgery/stats.pdf
                    if self.emm:
                        if self.fwd > 0:
                            zg2_ = torch.mean(torch.stack(z_fwd, dim=0), dim=0)
                            ze2_ = self.embedding[idx]
                            zdf2_ = zg2_ - ze2_
                            zic2_ = self.alpha * zdf2_
                            zvr2_ = self.embedding_var[idx]
                            zvr_ = torch.mean((1.0 - self.alpha) * (zvr2_ + zdf2_ * zic2_), dim=1, keepdim=True)
                            ze_ = ze2_ + zic2_
                        else: #EMM or EMM+ASM
                            zvr_ = self.embedding_var[idx]
                            ze_ = self.embedding[idx]
                        ze0_ = ze_
                        ze1_ = ze_
                    elif self.fwd > 0:
                        ze0_ = z_fwd[0]
                        ze1_ = z_fwd[1] if self.fwd == 2 else z_fwd[0]
                        zvr_ = self.embedding_var[idx]
                    else:
                        raise Exception("Not Valid Combo")

                    zdf0_ = zg0_ - ze0_
                    zic0_ = self.alpha * zdf0_
                    sigma0_ = torch.mean((1.0 - self.alpha) * (zvr_ + zdf0_ * zic0_), dim=1, keepdim=True)

                    zdf1_ = zg1_ - ze1_
                    zic1_ = self.alpha * zdf1_
                    sigma1_ = torch.mean((1.0 - self.alpha) * (zvr_ + zdf1_ * zic1_), dim=1, keepdim=True)
                    
                    sigma_ = (sigma0_+sigma1_)/2.0
                    zic_ = (zic0_+zic1_)/2.0

                    # https://openaccess.thecvf.com/content/WACV2024/papers/Khoshsirat_Improving_Normalization_With_the_James-Stein_Estimator_WACV_2024_paper.pdf
                    norm0_ = torch.linalg.vector_norm(zg0_-ze0_, dim=1, keepdim=True)**2
                    norm1_ = torch.linalg.vector_norm(zg1_-ze1_, dim=1, keepdim=True)**2
                    
                    if self.emm_v == 3:
                        sigma__ = (self.prd_width-2.0)*(torch.mean((0.5*(zg0_-zg1_))**2, dim=1, keepdim=True))
                    if self.emm_v == 1:
                        sigma__ = (self.prd_width-2.0)*(torch.mean((0.5*(zg0_-zg1_))**2))
                    if self.emm_v == 0:
                        sigma__ = (self.prd_width-2.0)*sigma_
                    
                    n0 = torch.maximum(1.0 - sigma__/norm0_, torch.tensor(0.0))
                    n1 = torch.maximum(1.0 - sigma__/norm1_, torch.tensor(0.0))

                    self.log_dict({"JS_n0_n1":0.5*(n0.mean() + n1.mean())})

                    if self.JS_INV:
                        zg0_ = (1.-n0)*zg0_ + n0*ze0_
                        zg1_ = (1.-n1)*zg1_ + n1*ze1_
                    else:
                        zg0_ = n0*zg0_ + (1.-n0)*ze0_
                        zg1_ = n1*zg1_ + (1.-n1)*ze1_
                    
                    if self.emm and self.fwd > 0:
                        self.embedding[idx] = ze_
                        self.embedding_var[idx] = zvr_
                    elif self.emm and self.asm:
                        self.embedding[idx] = ze_ + zic1_
                        self.embedding_var[idx] = sigma1_
                    elif self.emm:
                        self.embedding[idx] = ze_ + zic_
                        self.embedding_var[idx] = sigma_
                    elif self.fwd > 0:
                        self.embedding_var[idx] = sigma_
                    else:
                        raise Exception("Not Valid Combo")
                        

            if self.ema_v2: #For EMA 2.0
                raise NotImplementedError("ema v2")
                # n = cosine_schedule(self.global_step, 
                #                     self.trainer.estimated_stepping_batches, 
                #                     self.n0, self.n1)            
                # zg_ = 0.5*(zg0_+zg1_)
                # if self.first_epoch:
                #     self.embedding[idx] = zg_.detach()
                # else:                    
                #     if self.fwd > 0 and self.emm:
                #         ze_ = (self.embedding[idx] + zg2_)/2.0
                #     elif self.emm:
                #         ze_ = self.embedding[idx]
                #     elif self.fwd > 0:
                #         ze_ = zg2_

                #     #1 means only previous, 0 means only current
                #     zg0_ = (n)*zg0_ + (1.-n)*ze_
                #     zg1_ = (n)*zg1_ + (1.-n)*ze_

                #     if self.alpha < 1.0:
                #         self.embedding[idx] = (self.alpha)*zg_ + (1.-self.alpha)*ze_
                #     else:
                #         self.embedding[idx] = zg_
            
            z = [zg1_, zg0_]
            if views > self.fwd + 2:
                zg_ = 0.5*(zg0_+zg1_)
                z.extend([zg_ for _ in range(views-2-self.fwd)])
            
            if self.asm:
                p = p[:1]
                z = z[:1]
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
            # self.embedding_diff = torch.empty((N, self.prd_width),
            #                             dtype=torch.float16,
            #                             device=self.device)
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
        params_weight_decay, params_no_weight_decay = get_weight_decay_parameters(
                    [self.backbone, self.prediction_head, self.projection_head]
                )
        optimizer = SGD(
            [
                {   "name": "params_weight_decay", 
                    "params": params_weight_decay,
                },
                {
                    "name": "params_no_weight_decay", 
                    "params": params_no_weight_decay,
                    "weight_decay": 0.0,
                },
                # {
                #     "name": "proj",
                #     "params": self.projection_head.parameters(),                    
                # },     
                {
                    "name": "online_classifier",
                    "params": self.online_classifier.parameters(),
                    "weight_decay": 0.0,
                    "lr": 0.1
                },
            ],            
            lr=self.lr, #* self.batch_size_per_device * self.trainer.world_size / 256,
            momentum=0.9,
            weight_decay=self.decay,
        )         
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=int(
                      self.trainer.estimated_stepping_batches
                    / self.trainer.max_epochs
                    * self.warmup
                ),
                end_value=0.001,
                max_epochs=int(self.trainer.estimated_stepping_batches),
            ),
            "interval": "step",
        }
        # scheduler = {"scheduler": ConstantLR(optimizer=optimizer,factor=1.0,total_iters=1),"interval": "step",}
                     
        return [optimizer], [scheduler]

# For ResNet50 we adjust crop scales as recommended by the authors:
# https://github.com/facebookresearch/dino#resnet-50-and-other-convnets-trainings
# transform = DINOTransform(global_crop_scale=(0.14, 1), local_crop_scale=(0.05, 0.14), n_local_views=n_local_views)
def train_transform(size, scale=(0.08, 1.0), NORMALIZE=IMAGENET_NORMALIZE):
    return T.Compose([
                    T.RandomResizedCrop(size, scale=scale),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(mean=NORMALIZE["mean"], std=NORMALIZE["std"]),
                ])
val_identity  = lambda size, NORMALIZE: T.Compose([
                    T.Resize(size), T.ToTensor(),
                    T.Normalize(mean=NORMALIZE["mean"], std=NORMALIZE["std"]),
                ])
val_transform = T.Compose([
                    T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize(mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]),
                ])

CIFAR100_NORMALIZE  = {'mean':(0.5071, 0.4867, 0.4408), 'std':(0.2675, 0.2565, 0.2761)}
CIFAR10_NORMALIZE   = {'mean':(0.4914, 0.4822, 0.4465), 'std':(0.2470, 0.2435, 0.2616)}
TINYIMAGE_NORMALIZE = {'mean':(0.4802, 0.4481, 0.3975), 'std':(0.2302, 0.2265, 0.2262)}
STL10_NORMALIZE     = {'mean':(0.4408, 0.4279, 0.3867), 'std':(0.2682, 0.2610, 0.2686)}

transforms = {
"Cifar10":      DINOTransform(global_crop_size=32,
                            global_crop_scale=(0.20, 1.0),
                            n_local_views=0,
                            gaussian_blur=(0.5, 0.0, 0.0),
                            normalize=CIFAR10_NORMALIZE),

"Cifar100-asm": JSREPATransform(global_crop_size=32,
                            global_crop_scale=(0.14, 1.0),
                            weak_crop_scale=(0.14, 1.0),
                            n_global_views=1,
                            n_weak_views=1,                         
                            n_local_views=0,
                            gaussian_blur=(0.5, 0.0, 0.0),
                            normalize=CIFAR100_NORMALIZE),
"Cifar100-weak":JSREPATransform(global_crop_size=32,
                            global_crop_scale=(0.14, 1.0),
                            weak_crop_scale=(0.14, 1.0),
                            n_global_views=2,
                            n_weak_views=1,
                            n_local_views=0,
                            gaussian_blur=(0.5, 0.0, 0.0),
                            normalize=CIFAR100_NORMALIZE),
"Cifar100-2":     DINOTransform(global_crop_size=32,
                            global_crop_scale=(0.14, 1.0),
                            n_local_views=0,
                            gaussian_blur=(0.5, 0.0, 0.0),
                            normalize=CIFAR100_NORMALIZE),
"Cifar100-4":   DINOTransform(global_crop_size=32,
                            global_crop_scale=(0.14, 1.0),
                            n_local_views=6,
                            local_crop_size=32,
                            local_crop_scale=(0.14, 1.0),
                            gaussian_blur=(0.5, 0.0, 0.0),
                            normalize=CIFAR100_NORMALIZE),

"Tiny-asm":     JSREPATransform(global_crop_size=64,                    
                            global_crop_scale=(0.14, 1.0),
                            weak_crop_scale=(0.14, 1.0),
                            n_global_views=1,
                            n_weak_views=1,                         
                            n_local_views=0,
                            gaussian_blur=(0.5, 0.0, 0.0),
                            normalize=TINYIMAGE_NORMALIZE),
"Tiny-weak":    JSREPATransform(global_crop_size=64,                    
                            global_crop_scale=(0.20, 1.0),
                            weak_crop_scale=(0.20, 1.0),
                            n_global_views=2,
                            n_weak_views=1,
                            n_local_views=0,
                            gaussian_blur=(0.5, 0.0, 0.0),
                            normalize=TINYIMAGE_NORMALIZE),
"Tiny-2":       DINOTransform(global_crop_size=64,                          
                            global_crop_scale=(0.14, 1.0),
                            n_local_views=0,
                            gaussian_blur=(0.5, 0.0, 0.0),
                            normalize=TINYIMAGE_NORMALIZE),

"Tiny-4":       DINOTransform(global_crop_size=64,
                            global_crop_scale=(0.20, 1.0),
                            n_local_views=2,
                            local_crop_size=64,
                            local_crop_scale=(0.20, 1.0),
                            gaussian_blur=(0.5, 0.0, 0.0),
                            normalize=TINYIMAGE_NORMALIZE),
"Tiny-8":       DINOTransform(global_crop_size=64,
                            global_crop_scale=(0.2, 1.0),
                            local_crop_size=32,
                            local_crop_scale=(0.05, 0.2),
                            gaussian_blur=(0.5, 0.0, 0.0),
                            normalize=TINYIMAGE_NORMALIZE),

"STL-2":        DINOTransform(global_crop_size=96,
                            global_crop_scale=(0.20, 1.0),
                            n_local_views=0,
                            gaussian_blur=(0.5, 0.0, 0.0),
                            normalize=STL10_NORMALIZE),

"Im100-8":      DINOTransform(global_crop_scale=(0.14, 1.00),
                            local_crop_scale=(0.05, 0.14)),
"Im100-2-20":   DINOTransform(global_crop_scale=(0.20, 1.0),
                            n_local_views=0),
"Im100-2-14":   DINOTransform(global_crop_scale=(0.14, 1.0),
                            n_local_views=0),
"Im100-2-08":   DINOTransform(global_crop_scale=(0.08, 1.0),
                            n_local_views=0),
"Im100-2-05":   DINOTransform(global_crop_scale=(0.05, 1.0),
                            n_local_views=0),
"Im100-weak":   JSREPATransform(global_crop_scale=(0.08, 1.0),
                            weak_crop_scale=(0.08, 1.0),
                            n_global_views=2,
                            n_weak_views=1,
                            n_local_views=0),


"Im1k-8":       DINOTransform(global_crop_scale=(0.14, 1.00),
                            local_crop_scale =(0.05, 0.14)),
"Im1k-2":       DINOTransform(global_crop_scale=(0.14, 1.00),
                            n_local_views=0),
}

train_transforms = {
"Cifar10":   T.Compose([                    
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(mean=CIFAR10_NORMALIZE["mean"], std=CIFAR10_NORMALIZE["std"]),
                ]),
"Cifar100":  T.Compose([                    
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(mean=CIFAR100_NORMALIZE["mean"], std=CIFAR100_NORMALIZE["std"]),
                ]),
"Tiny":      T.Compose([
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(mean=TINYIMAGE_NORMALIZE["mean"], std=TINYIMAGE_NORMALIZE["std"]),
                ]),
# "Tiny":      train_transform(64, NORMALIZE=TINYIMAGE_NORMALIZE),
"STL":       train_transform(96, NORMALIZE=STL10_NORMALIZE),
"Im100":     train_transform(224),
"Im1k":      train_transform(224),
}

val_transforms = {
"Cifar10":   val_identity(32, CIFAR100_NORMALIZE),
"Cifar100":  val_identity(32, CIFAR100_NORMALIZE),
"Tiny":      val_identity(64, TINYIMAGE_NORMALIZE),
"STL":       val_identity(96, STL10_NORMALIZE),
"Im100":     val_transform,
"Im1k":      val_transform,
}
