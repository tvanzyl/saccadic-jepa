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
from lightly.loss.koleo_loss import KoLeoLoss
from lightly.models.utils import (
    get_weight_decay_parameters,
)

from lightly.transforms import DINOTransform
from lightly.utils.benchmarking import OnlineLinearClassifier
from lightly.utils.scheduler import CosineWarmupScheduler, cosine_schedule
from lightly.utils.debug import std_of_l2_normalized

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

class StandardiseLayer(nn.Module):
    def __init__(self, temp:float=1.0, dim:int=0):
        super(StandardiseLayer, self).__init__()
        self.dim = dim        
        self.temp= temp
    
    def forward(self, x: Tensor) -> Tensor:    
        return (x - torch.mean(x, dim=self.dim, keepdim=True))/(torch.var(x, dim=self.dim, keepdim=True)*self.temp)

class CenteringLayer(nn.Module):
    def __init__(self, dim:int=0):
        super(CenteringLayer, self).__init__()
        self.dim = dim        
    
    def forward(self, x: Tensor) -> Tensor:                
        return x - torch.mean(x, dim=self.dim, keepdim=True)

class ScalingLayer(nn.Module):
    def __init__(self, temp:float=1.0, dim:int=0):
        super(ScalingLayer, self).__init__()
        self.dim = dim
        self.temp = temp
    
    def forward(self, x: Tensor) -> Tensor:    
        return x/(torch.var(x, dim=self.dim, keepdim=True)*self.temp)

class BiasLayer(nn.Module):
    def __init__(self, size:int):
        super(BiasLayer, self).__init__()
        self.bias = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        return x + self.bias

class LinearEnsemble(nn.Module):
    def __init__(self, input_dim, output_dim, num_models):
        super(LinearEnsemble, self).__init__()
        self.num_models = num_models
        self.models = nn.ModuleList([
            nn.Linear(input_dim, output_dim) for _ in range(num_models)
        ])

    def forward(self, x):
        # Collect outputs from each individual linear layer
        outputs = [model(x) for model in self.models]

        # Combine the outputs (e.g., by averaging them)
        # Stacking and then taking the mean across the model dimension
        stacked_outputs = torch.stack(outputs, dim=0)
        ensemble_output = torch.mean(stacked_outputs, dim=0)
        return ensemble_output


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
                 warmup: int = 0,
                 backbone:str = "resnet-50",
                 n_local_views:int = 6,
                 lr:float = 0.15,
                 decay:float=1e-4,                                  
                 momentum_head:bool=False,
                 identity_head:bool=False,
                 no_projection_head:bool=False,                 
                 alpha:float = 0.80, gamma:float = 0.50,
                 cut:float = 9.0,                 
                 prd_width:int = 256,
                 prj_depth:int = 2,
                 prj_width:int = 2048,
                 L2:bool=False,
                 no_ReLU_buttress:bool=False,
                 no_prediction_head:bool=False,
                 JS:bool=False, 
                 no_bias:bool=False,
                 emm:bool=False, emm_v:int=0,
                 fwd:int=0,
                 asm:bool=False,                 
                 nn_init:str="fan-in",                 
                 end_value:float=0.001) -> None:
        super().__init__()
        self.save_hyperparameters('batch_size_per_device',
                                  'num_classes', 'warmup',
                                  'backbone',
                                  'n_local_views',
                                  'lr',
                                  'decay',
                                  'JS',
                                  'momentum_head',
                                  'identity_head',
                                  'no_projection_head',
                                  'no_prediction_head',                                  
                                  'alpha', 'gamma',                                  
                                  'cut','prd_width', 
                                  "prj_depth", "prj_width",
                                  'L2',
                                  'no_ReLU_buttress',
                                  'emm', 'emm_v', 
                                  'no_bias',
                                  'fwd',
                                  'asm',                                   
                                  'nn_init',                                  
                                  'end_value')
        self.warmup = warmup
        self.lr = lr
        self.decay = decay
        self.batch_size_per_device = batch_size_per_device                
        self.JS = JS        
        self.emm = emm
        self.emm_v = emm_v
        self.fwd = fwd
        self.asm = asm
        self.momentum_head = momentum_head                
        self.alpha = alpha
        self.gamma = gamma               
        self.no_ReLU_buttress = no_ReLU_buttress
        self.end_value = end_value

        if identity_head and momentum_head:
            raise Exception("Invalid Arguments, can't select identity and momentum")
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
                                   nn.LeakyReLU(),
                                   nn.Linear(prj_width, prj_width, False),
                                   nn.BatchNorm1d(prj_width),
                                   nn.LeakyReLU(),
                                   nn.Linear(prj_width, prj_width),]
            elif prj_depth == 1:
                projection_head = [nn.Linear(emb_width, prj_width, False),
                                   nn.BatchNorm1d(prj_width),
                                   nn.LeakyReLU(),
                                   nn.Linear(prj_width, prj_width),]
            elif prj_depth == 0:
                projection_head = [nn.Linear(emb_width, prj_width),]
            else:
                raise NotImplementedError("Selected Prediction Depth Not Supported")
                
            if L2:
                projection_head.insert(0, L2NormalizationLayer())                

            self.projection_head = nn.Sequential(          
                                    *projection_head
                                )
        
        if no_prediction_head:
            self.prediction_head = nn.AdaptiveAvgPool1d(self.prd_width)
        else:
            self.prediction_head = nn.Linear(prj_width, self.prd_width, False)

            if nn_init == "rand-in":
                bound_w = 1 / (math.sqrt(self.prediction_head.weight.size(1))*9)
            elif nn_init == "rand-out":
                # https://arxiv.org/pdf/2406.16468 (Cut Init)
                bound_w = 1 / math.sqrt(self.prediction_head.weight.size(0)*9)
            elif nn_init == "fan-in":
                bound_w = 1 / math.sqrt(self.prediction_head.weight.size(1))
            elif nn_init == "fan-out":
                bound_w = 1 / math.sqrt(self.prediction_head.weight.size(0))
            elif nn_init == "he-in":
                bound_w = math.sqrt(3) / math.sqrt(self.prediction_head.weight.size(1))
            elif nn_init == "he-out":
                bound_w = math.sqrt(3) / math.sqrt(self.prediction_head.weight.size(0))
            elif nn_init == "xavier":
                bound_w = math.sqrt(6) / math.sqrt(self.prediction_head.weight.size(0) + self.prediction_head.weight.size(1))
            
            # nn.init.normal_(self.prediction_head.weight, 0, bound_w)
            nn.init.orthogonal_(self.prediction_head.weight)

        #Use Batchnorm none-affine for centering
        if no_bias:
            self.buttress = nn.BatchNorm1d(prj_width, affine=False, track_running_stats=False)
        else:
            biaslayer = BiasLayer(prj_width)
            nn.init.normal_(biaslayer.bias, 0, bound_w)
            self.buttress =  nn.Sequential(            
                                    nn.BatchNorm1d(prj_width, affine=False, track_running_stats=False),                                    
                                    biaslayer)

        if identity_head:
            if prj_width == prd_width:
                self.merge_head = nn.Identity()
            elif prj_width > prd_width:
                #Identity matrix hack for if requires dimensionality reduction
                self.merge_head = nn.AdaptiveAvgPool1d(self.prd_width)
                #Rescale ???
            else:
                raise NotImplementedError("Invalid Arguments, can't select prd width larger than prj width")
        else:            
            self.merge_head = nn.Linear(prj_width, self.prd_width, False)
            if no_prediction_head:
                nn.init.normal_(self.merge_head.weight, 0, bound_w)
                # nn.init.orthogonal_(self.merge_head.weight)
            else:
                self.merge_head.weight.data = self.prediction_head.weight.data.clone()
        
        if not no_prediction_head:            
            self.prediction_head.weight.data.div_(cut)
        
        if not no_ReLU_buttress:
            self.prediction_head = nn.Sequential(                                       
                                nn.ReLU(),                                
                                self.prediction_head,
                            )            
            self.merge_head = nn.Sequential(                                    
                                nn.ReLU(),
                                self.merge_head,
                            )
        
        self.criterion = NegativeCosineSimilarity()        

        self.online_classifier = OnlineLinearClassifier(feature_dim=emb_width, num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def forward_student(self, x: List[Tensor], idx: Tensor) -> Tensor:
        views = len(x)

        # Two globals
        f = [self.backbone( x_ ).flatten(start_dim=1) for x_ in x[:2]]        
        
        if views > self.fwd + 2: # MultiCrops
            f.extend([self.backbone( x_ ).flatten(start_dim=1) for x_ in x[self.fwd+2:]])
        b = [self.projection_head( f_ ) for f_ in f]
        g = [self.buttress( b_ ) for b_ in b]
        p = [self.prediction_head( b_ ) for b_ in b]
        
        with torch.no_grad(): 
            self.log_dict({"f_quality":std_of_l2_normalized(f[0])})
            self.log_dict({"f_mean":torch.mean(f[0])})
            self.log_dict({"f_var":torch.var(f[0])})
            self.log_dict({"f_sharp":torch.mean(f[0])/torch.var(f[0])})
            self.log_dict({"b_mean":torch.mean(b[0])})
            self.log_dict({"b_var":torch.var(b[0])})
            self.log_dict({"b_sharp":torch.mean(b[0])/torch.var(b[0])})            

            # Fwds Only
            if self.fwd > 0:            
                f_fwd = [self.backbone( x_ ).flatten(start_dim=1) for x_ in x[2:self.fwd+2]]
                b_fwd = [self.projection_head( f_ ) for f_ in f_fwd]
                g_fwd = [self.buttress( b_ ) for b_ in b_fwd]
                z_fwd = [self.merge_head( g_ ) for g_ in g_fwd]

            z = [self.merge_head( g_.detach() ) for g_ in g]
            zg0_ = z[0]
            zg1_ = z[1]

            if self.JS: # For James-Stein
                if self.first_epoch:
                    self.embedding[idx] = 0.5*(zg0_+zg1_)
                    if self.emm_v == 6 or self.emm_v == 5:
                        self.embedding_var[idx] = (0.5*(zg0_-zg1_))**2.0
                    elif self.emm_v <= 2:
                        self.embedding_var[idx] = torch.mean((0.5*(zg0_-zg1_))**2.0, dim=1, keepdim=True)
                else:
                    # EWM-A/V https://fanf2.user.srcf.net/hermes/doc/antiforgery/stats.pdf
                    if self.emm:
                        if self.fwd > 0:
                            zmean0_ = torch.mean(torch.stack(z_fwd, dim=0), dim=0)
                            zmean_ = self.embedding[idx]
                            zdiff_ = zmean0_ - zmean_
                            zincr_ = self.alpha * zdiff_
                            zmean_ = zmean_ + zincr_
                            self.embedding[idx] = zmean_
                        else: #EMM or EMM+ASM
                            zmean_ = self.embedding[idx]                            
                    elif self.fwd > 0: #Use forwards as the mean
                        # zmean_ = z_fwd[0]
                        if self.emm_v == 7:
                            zmean_ = z_fwd[0]
                        else:
                            zmean_ = torch.mean(torch.stack(z_fwd, dim=0), dim=0)
                    else: 
                        raise NotImplementedError()                    
                    
                    zdiff0_ = zg0_  - zmean_
                    zdiff1_ = zg1_  - zmean_

                    if self.emm_v == 7:
                        zmeanz_ = z_fwd[1]
                        sigma_ = self.gamma*((zg0_-zmeanz_)**2.0 + (zg1_-zmeanz_)**2.0)
                    elif self.emm_v == 6:
                        sigma_ = self.embedding_var[idx]
                        zincr0_ = self.gamma * zdiff0_
                        zincr1_ = self.gamma * zdiff1_
                        sigma_  = (1.0 - self.gamma) * (sigma_ + ((zdiff0_*zincr0_)+
                                                                  (zdiff1_*zincr1_))/2.0)
                        self.embedding_var[idx] = sigma_
                    elif self.emm_v == 5:
                        zmeanz_ = torch.mean(torch.stack(z, dim=0), dim=0)
                        zdiff2_ = z_fwd[0]  - zmeanz_
                        zdiff3_ = z_fwd[1]  - zmeanz_
                        sigma_ = self.embedding_var[idx]
                        zincr2_ = self.gamma * zdiff2_
                        zincr3_ = self.gamma * zdiff3_
                        sigma_  = (1.0 - self.gamma) * (sigma_ + ((zdiff2_*zincr2_)+
                                                                  (zdiff3_*zincr3_))/2.0)
                        self.embedding_var[idx] = sigma_
                    elif self.emm_v == 4:
                        zmeanz_ = torch.mean(torch.stack(z, dim=0), dim=0)
                        sigma_ = self.gamma*((z_fwd[0]-zmeanz_)**2.0 + (z_fwd[1]-zmeanz_)**2.0)
                    elif self.emm_v == 3:
                        sigma_ = torch.mean(((zg0_-zg1_)*self.gamma)**2.0)
                    elif self.emm_v == 2:
                        sigma_ = self.embedding_var[idx]
                        zincr0_ = self.gamma * zdiff0_
                        zincr1_ = self.gamma * zdiff1_
                        sigma_ = torch.mean((1.0 - self.gamma) * (sigma_ + ((zdiff0_*zincr0_)+
                                                                            (zdiff1_*zincr1_))/2.0), dim=1, keepdim=True)
                        self.embedding_var[idx] = sigma_
                        sigma_ = 0.5*sigma_ + 0.5*(zg0_-zg1_)**2.0
                    elif self.emm_v == 1:
                        sigma_ = self.embedding_var[idx]
                        zincr0_ = self.gamma * zdiff0_
                        zincr1_ = self.gamma * zdiff1_
                        sigma_ = torch.mean((1.0 - self.gamma) * (sigma_ + ((zdiff0_*zincr0_)+
                                                                            (zdiff1_*zincr1_))/2.0), dim=1, keepdim=True)
                        self.embedding_var[idx] = sigma_
                    else:
                        raise Exception("Not Valid EMM V")

                    # https://openaccess.thecvf.com/content/WACV2024/papers/Khoshsirat_Improving_Normalization_With_the_James-Stein_Estimator_WACV_2024_paper.pdf
                    norm0_ = torch.linalg.vector_norm((zdiff0_)/((sigma_**0.5)+1e-9), dim=1, keepdim=True)**2
                    norm1_ = torch.linalg.vector_norm((zdiff1_)/((sigma_**0.5)+1e-9), dim=1, keepdim=True)**2

                    n0 = torch.maximum(1.0 - (self.prd_width-2.0)/(norm0_+1e-9), torch.tensor(0.0))
                    n1 = torch.maximum(1.0 - (self.prd_width-2.0)/(norm1_+1e-9), torch.tensor(0.0))
                    
                    self.log_dict({"sigma":torch.mean(sigma_)})
                    self.log_dict({"zdiff":zdiff0_.mean()})                    
                    self.log_dict({"JS_n0_n1":n0.mean()})                    

                    zg0_ = n0*zg0_ + (1.-n0)*zmean_
                    zg1_ = n1*zg1_ + (1.-n1)*zmean_

                    if self.fwd > 0:
                        pass
                    elif self.asm: #TODO: Deal with ASM
                        # zic_ = (zincr0_+zincr1_)/2.0
                        zic_ = self.alpha*(zdiff0_+ zdiff1_)/2.0
                        self.embedding[idx] = zmean_ + zic_
                    elif self.emm:
                        # zic_ = (zincr0_+zincr1_)/2.0
                        zic_ = self.alpha*(zdiff0_+ zdiff1_)/2.0
                        self.embedding[idx] = zmean_ + zic_
                    else:
                        raise Exception("Not Valid Combo")
            
            z = [zg1_, zg0_]
            if views > self.fwd + 2:
                zg_ = 0.5*(zg0_+zg1_)
                z.extend([zg_ for _ in range(views-2-self.fwd)])
            
            if self.asm:
                p = p[:1]
                z = z[:1]
            assert len(p)==len(z)
        return f, p, z

    def on_train_epoch_end(self):
        if self.JS:
            self.first_epoch = False
        return super().on_train_epoch_end()

    def on_train_start(self):                
        if self.JS:
            self.first_epoch = True            
            N = len(self.trainer.train_dataloader.dataset)            
            if self.emm:
                self.embedding      = torch.empty((N, self.prd_width),
                                            dtype=torch.float16,
                                            device=self.device)
            if self.emm_v <= 2:
                self.embedding_var  = torch.zeros((N, 1),
                                        dtype=torch.float32,
                                        device=self.device)
            elif self.emm_v == 6 or self.emm_v == 5:
                self.embedding_var  = torch.zeros((N, self.prd_width),
                                        dtype=torch.float32,
                                        device=self.device)
        return super().on_train_start()

    def training_step(
        self, batch: Tuple[List[Tensor], Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        x, targets, idx = batch
        
        f, p, z = self.forward_student( x, idx )
        f0_ = f[0].detach()

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
                    [self.backbone, self.projection_head, self.prediction_head]                    
                )
        optimizer = SGD(
            [
                {   "name": "params_weight_decay", 
                    "params": params_weight_decay,
                },
                {   "name": "params_no_weight_decay", 
                    "params": params_no_weight_decay,
                    "weight_decay": 0.0,
                },                
                {   "name": "online_classifier",
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
                warmup_epochs=int(
                      self.trainer.estimated_stepping_batches
                    / self.trainer.max_epochs
                    * self.warmup
                ),
                end_value=self.end_value,
                max_epochs=int(self.trainer.estimated_stepping_batches),
            ),
            "interval": "step",
        }

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
                            global_crop_scale=(0.20, 1.0),
                            weak_crop_scale=(0.20, 1.0),
                            n_global_views=2,
                            n_weak_views=1,
                            n_local_views=0,
                            gaussian_blur=(0.5, 0.0, 0.0),
                            normalize=CIFAR100_NORMALIZE),
"Cifar100-weak-2":JSREPATransform(global_crop_size=32,
                            global_crop_scale=(0.14, 1.0),
                            weak_crop_scale=(0.14, 1.0),
                            n_global_views=2,
                            n_weak_views=2,
                            n_local_views=0,
                            gaussian_blur=(0.5, 0.1, 0.0),
                            normalize=CIFAR100_NORMALIZE),
"Cifar100-2":   DINOTransform(global_crop_size=32,
                            global_crop_scale=(0.20, 1.0),
                            n_local_views=0,
                            gaussian_blur=(0.0, 0.0, 0.0),                            
                            normalize=CIFAR100_NORMALIZE),
"Cifar100-4":   DINOTransform(global_crop_size=32,
                            global_crop_scale=(0.20, 1.0),
                            n_local_views=6,
                            local_crop_size=32,
                            local_crop_scale=(0.20, 1.0),
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
