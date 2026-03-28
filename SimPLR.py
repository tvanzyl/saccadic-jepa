import math
import copy
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import SGD

from torchvision.models import resnet50, resnet34 , resnet18
from torchvision.transforms import v2 as T

from pytorch_lightning import LightningModule

from lightly.transforms import DINOTransform
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.loss import NegativeCosineSimilarity
from lightly.models.utils import (
    get_weight_decay_parameters,
    deactivate_requires_grad, 
    update_momentum
)
from lightly.models._momentum import _do_momentum_update

from lightly.utils.benchmarking import OnlineLinearClassifier
from lightly.utils.scheduler import CosineWarmupScheduler, cosine_schedule
from lightly.utils.debug import std_of_l2_normalized

class SIGReg(torch.nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), 256, device="cuda")
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()

class BiasLayer(nn.Module):
    def __init__(self, size:int, fan_in_size:int):
        super(BiasLayer, self).__init__()
        self.bias = nn.Parameter(torch.zeros(size))
        bound_w = 1 / math.sqrt(fan_in_size)
        nn.init.normal_(self.bias, 0, bound_w)

    def forward(self, x):
        return x + self.bias

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

#https://proceedings.mlr.press/v202/garrido23a/garrido23a.pdf
@torch.no_grad()
def effective_rank(embeddings, eps=1e-9):
    """
    Computes the effective rank and condition number of a batch of embeddings.
    
    Args:
        embeddings: Tensor of shape [Batch_Size, Dimension]
        eps: Small value for numerical stability
        
    Returns:
        effective_rank: Float representing the continuous dimensionality.
        condition_number: Float representing the ratio of max to min variance.
    """
    # 1. Mean-center the embeddings
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    
    # 2. Compute SVD
    # full_matrices=False is critical for performance when Batch_Size > Dimension
    _, S, _ = torch.linalg.svd(centered, full_matrices=False)
    
    # 3. Convert singular values to eigenvalues (variance along principal components)
    eigenvalues = (S ** 2) / (centered.size(0) - 1)
    
    # 4. Normalize to a probability distribution
    p = eigenvalues / (eigenvalues.sum() + eps)
    
    # 5. Compute Shannon Entropy and Effective Rank
    entropy = -torch.sum(p * torch.log(p + eps))
    effective_rank = torch.exp(entropy).item()
    
    # 6. Compute Condition Number (Optional but highly recommended)
    # A spiking condition number is the earliest warning sign of collapse.
    condition_number = (eigenvalues[0] / (eigenvalues[-1] + eps)).item()
    
    return effective_rank, condition_number

class SIGReg(torch.nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), 256, device="cuda")
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()

def backbones(name):
    if name in ["resnetjie-9","resnetjie-18"]:
        resnet = {"resnetjie-9" :resnet18,
                  "resnetjie-18":resnet18}[name]()
        emb_width = resnet.fc.in_features
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
        resnet.maxpool = nn.Sequential()
        resnet.fc = nn.Identity()
    elif name in ["resnet-18", "resnet-34", "resnet-50"]: 
        resnet = {"resnet-18":resnet18, 
                  "resnet-34":resnet34, 
                  "resnet-50":resnet50}[name](zero_init_residual=(name!="resnet-18"))
        emb_width = resnet.fc.in_features
        resnet.fc = nn.Identity()
    else:
        raise NotImplemented("Backbone Not Supported")
    return resnet, emb_width

class SimPLR(LightningModule):
    def __init__(self, batch_size_per_device: int,                  
                 num_classes: int, 
                 warmup: int = 0,
                 backbone:str = "resnet-50",                 
                 lr:float = 0.5, linear_lr:float = 0.1,
                 decay:float=1e-5,                                  
                 momentum_head:bool=False,
                 identity_head:bool=False,
                 no_projection_head:bool=False,
                 alpha:float = 1.00, lambd:float = 0.00,
                 cut:float = 0.0,
                 prd_width:int = 256,
                 prj_depth:int = 2,
                 prj_width:int = 2048,                 
                 no_buttress:bool=False,
                 no_ReLU_buttress:bool=False,
                 no_student_head:bool=False,
                 JS:bool=False, 
                 no_bias:bool=False,
                 ema:bool=False, emm_v:int=8, var:float=0.1,                 
                 momentum_butt:bool=False) -> None:
        super().__init__()
        self.save_hyperparameters('batch_size_per_device',
                                  'num_classes', 'warmup',
                                  'backbone',                                  
                                  'lr', 'decay', 'JS',
                                  'momentum_head',
                                  'identity_head',
                                  'no_projection_head',
                                  'no_student_head',
                                  'alpha', 'lambd',
                                  'cut','prd_width', 
                                  "prj_depth", "prj_width",
                                  'no_buttress',
                                  'no_ReLU_buttress',
                                  'ema', 'emm_v', 'var',
                                  'no_bias',                                  
                                  'momentum_butt')
        self.warmup = warmup
        self.lr = lr
        self.linear_lr = linear_lr
        self.decay = decay
        self.batch_size_per_device = batch_size_per_device
        self.JS = JS        
        self.ema = ema
        self.emm_v = emm_v
        self.var = torch.tensor(var, device=self.device, requires_grad=False)
        self.momentum_head = momentum_head        
        self.alpha_alpha = alpha
        self.lambd = lambd
        self.momentum_butt = momentum_butt
        
        self.prd_width = prd_width

        if identity_head and momentum_head:
            raise Exception("Invalid Arguments, can't select identity and momentum")
        
        self.backbone, self.emb_width = backbones(backbone)
        emb_width = self.emb_width

        if self.ema:
            self.teacher_backbone = copy.deepcopy(self.backbone)
            deactivate_requires_grad(self.teacher_backbone)

        if no_projection_head:
            prj_width = emb_width        
        
        self.projection_head = nn.Sequential()
        if no_projection_head:   
            self.projection_head.append(nn.Identity())
            prj_width = emb_width
        else:            
            self.projection_head.extend([nn.Linear(emb_width, prj_width, bias=False),])
            for i in range(prj_depth):
                self.projection_head.extend(
                                        [nn.BatchNorm1d(prj_width),
                                         nn.ReLU(),
                                         nn.Linear(prj_width, prj_width, bias=(i==prj_depth-1))]
                                        )
        
        
        if self.ema:
            self.teacher_projection_head = copy.deepcopy(self.projection_head)
            deactivate_requires_grad(self.teacher_projection_head)

        #Use Batchnorm non-affine for centering
        if no_buttress:
            self.buttress = nn.Identity()
        else:
            self.buttress = nn.BatchNorm1d(prj_width,
                                        affine=False,
                                        momentum=0.9)
        if identity_head:
            if prj_width > prd_width:
                #Identity matrix hack for if requires dimensionality reduction
                teacher_head = nn.AdaptiveAvgPool1d(prd_width)
            elif prj_width == prd_width:
                teacher_head = nn.Identity()
            else:
                raise NotImplementedError("Invalid Arguments, can't select prd width larger than prj width")
        else:
            teacher_head = nn.Linear(prj_width, self.prd_width, False)
            nn.init.orthogonal_(teacher_head.weight)
        if no_bias:        
            self.teacher_head = nn.Sequential(teacher_head)
        else:
            self.teacher_head = nn.Sequential(
                                            teacher_head,
                                            BiasLayer(prd_width, prj_width),
                                        )

        if no_student_head:
            student_head = nn.AdaptiveAvgPool1d(prd_width)
        else:
            student_head = nn.Linear(prj_width, prd_width, False)
            if not identity_head:
                if cut == 0.0:
                    cut = (teacher_head.weight.data.var()/student_head.weight.data.var())**0.5
                    print(f"Cut: {cut}")
                student_head.weight.data = teacher_head.weight.data.clone()
            if cut > 0.0: # https://arxiv.org/pdf/2406.16468 (Cut Init)
                student_head.weight.data.div_(cut)
        self.student_head = nn.Sequential(student_head)

        if not no_ReLU_buttress:
            self.student_head.insert(0, nn.ReLU())
            self.teacher_head.insert(0, nn.ReLU())
        
        deactivate_requires_grad(self.teacher_head)

        self.var_head = nn.Sequential()
        if self.emm_v == 8 and self.JS:
            self.var_head.extend(
                                [nn.Linear(emb_width, prj_width),
                                 nn.SiLU(),
                                 nn.Linear(prj_width, prd_width),
                                 nn.Softplus()]
                            )
        
        self.online_classifier = OnlineLinearClassifier(feature_dim=emb_width, num_classes=num_classes)
        
        self.criterion = NegativeCosineSimilarity()
        self.regularisation = SIGReg()
        self.var_crt = nn.GaussianNLLLoss()

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def forward_student(self, x: List[Tensor], idx: Tensor) -> Tensor:
        views = len(x)
        
        # Two globals
        h = [self.backbone( x_ ) for x_ in x[:2]]
        h0_ = h[0].detach()
        z = [self.projection_head( h_ ) for h_ in h]
        p = [self.student_head( z_ ) for z_ in z]
        
        if views > 2: # MultiCrops
            h_multi = [self.backbone( x_ ).flatten(start_dim=1) for x_ in x[2:]]
            z_multi = [self.projection_head( h_ ) for h_ in h_multi]
            p.extend([self.student_head( z_ ) for z_ in z_multi])        
        
        if self.emm_v == 8 and self.JS:
            vars = [self.var_head( h_.detach() ) for h_ in h]
        else:
            vars = None       

        # with torch.no_grad():
        if self.momentum_butt:
            # Use Momentum Statistics
            self.buttress(torch.cat(z))
            self.buttress.train(False)

        if self.ema:
            ht = [self.teacher_backbone( x_ ).flatten(start_dim=1) for x_ in x[:2]]
            zt = [self.teacher_projection_head( h_ ) for h_ in ht]
        else:
            zt = z
        bt = [self.buttress( z_.detach() ) for z_ in zt]
        qo = [self.teacher_head( b_ ) for b_ in bt]
        q0_ = qo[0]
        q1_ = qo[1]
        
        if self.momentum_butt:
            self.buttress.train(True)

        if self.JS: # For James-Stein
            if self.current_epoch == 0:
                self.embedding[idx] = (0.5*(q0_+q1_)).to(torch.float32)
                if self.emm_v in [7,8]:
                    self.var  = torch.mean( (q0_-q1_)**2.0 )
            else:
                #EMM, EWM-A/V https://fanf2.user.srcf.net/hermes/doc/antiforgery/stats.pdf
                mean_ = self.embedding[idx]
                qdiff0_ = q0_  - mean_
                qdiff1_ = q1_  - mean_

                if self.emm_v == 9:
                    var_ = self.var
                elif self.emm_v == 8:
                    var_ = torch.mean( torch.stack(vars, dim=0).detach(), dim=0)
                elif self.emm_v == 7:                    
                    self.var =  torch.mean( (1.0-self.alpha)*self.var + self.alpha*qdiff0_.abs()*qdiff1_.abs() )
                    var_ = self.var 
                elif self.emm_v == 6:
                    var_ = torch.mean( qdiff0_.abs()*qdiff1_.abs() )
                else:
                    raise Exception("Not Valid EMM V")                    

                # https://openaccess.thecvf.com/content/WACV2024/papers/Khoshsirat_Improving_Normalization_With_the_James-Stein_Estimator_WACV_2024_paper.pdf
                norm0_ = torch.linalg.vector_norm((qdiff0_)/((var_**0.5)+1e-9), dim=1, keepdim=True)**2
                norm1_ = torch.linalg.vector_norm((qdiff1_)/((var_**0.5)+1e-9), dim=1, keepdim=True)**2

                n0 = torch.maximum(1.0 - (self.prd_width-2.0)/(norm0_+1e-9), torch.tensor(0.0))
                n1 = torch.maximum(1.0 - (self.prd_width-2.0)/(norm1_+1e-9), torch.tensor(0.0))

                self.log_dict({"JS_n0_n1":n0.mean()})
                self.log_dict({"var":torch.mean(var_)})

                q0_ = n0*q0_ + (1.-n0)*mean_
                q1_ = n1*q1_ + (1.-n1)*mean_                    
                
                incr_ = self.alpha*(qdiff0_+ qdiff1_)/2.0
                self.embedding[idx] = (mean_ + incr_).to(torch.float32)
            
        q  = [q1_, q0_]
        if views > 2:
            q_ = 0.5*(q0_+q1_)
            q.extend([q_ for _ in range(views-2)])
        assert len(p)==len(q)

        return h0_, p, q, z, qo, vars

    def on_train_start(self):
        if self.JS:
            N = len(self.trainer.train_dataloader.dataset)            
            self.embedding  = torch.empty((N, self.prd_width),
                                        # dtype=torch.float16,
                                        device=self.device,
                                        requires_grad=False)
        return super().on_train_start()

    def training_step(
        self, batch: Tuple[List[Tensor], Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        x, targets, idx = batch        
        
        self.alpha = cosine_schedule(self.global_step, self.trainer.estimated_stepping_batches, 0.99, self.alpha_alpha)

        momentum = cosine_schedule(self.global_step, self.trainer.estimated_stepping_batches, 0.996, 1)
        if self.ema: #These lines give us classical EMA 
            update_momentum(self.backbone, self.teacher_backbone, m=momentum)            
            update_momentum(self.projection_head, self.teacher_projection_head, m=momentum)
        if self.momentum_head:
            _do_momentum_update(self.teacher_head[-1].weight, self.student_head[-1].weight, momentum)

        h0_, p, q, z, qo, vars = self.forward_student( x, idx )

        loss = 0
        var_loss = 0        
        qomean_ = torch.mean(torch.stack(qo, dim=0), dim=0).detach()
        for xi in range(len(q)):
            p_ = p[xi]
            q_ = q[xi]
            loss += self.criterion( p_, q_ ) / len(q)

            if self.emm_v == 8 and self.JS:
                var_  = vars[xi]
                qo_   = qo[xi].detach()
                var_loss += self.var_crt(math.sqrt(2)*qomean_, math.sqrt(2)*qo_, var_)
                
        if self.lambd > 0.0:
            sigreg_loss = self.regularisation(torch.stack(z))
            loss = sigreg_loss*self.lambd + loss*(1.0-self.lambd)

            self.log_dict(
                {"sigreg_loss": sigreg_loss},            
                sync_dist=True,
                batch_size=len(targets),
            )
        
        self.log_dict(
            {"train_loss": loss},
            prog_bar=True,
            sync_dist=True,
            batch_size=len(targets),
        )
        self.log_dict(
            {"var_loss": var_loss}, 
            sync_dist=True, 
            batch_size=len(targets))

        # Online classification.
        cls_loss, cls_log = self.online_classifier.training_step(
            (h0_, targets), batch_idx
        )
        self.log_dict(cls_log, sync_dist=True, batch_size=len(targets))       

        self.log_dict(
            {"h_quality":std_of_l2_normalized(h0_)},
            sync_dist=True)
        self.log_dict(
            {"butt_e_rank":effective_rank(torch.cat(z).to(torch.float32))[0]},
            sync_dist=True)
        self.log_dict(
            {"student_e_rank":effective_rank(torch.cat(p).to(torch.float32))[0]},
            sync_dist=True)
        self.log_dict(
            {"teacher_e_rank":effective_rank(torch.cat(q).to(torch.float32))[0]},
            sync_dist=True)

        return loss + cls_loss + var_loss

    def validation_step(
        self, batch: Tuple[Tensor, Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        images, targets = batch[0], batch[1]
        features = self.forward(images).flatten(start_dim=1)
        cls_loss, cls_log = self.online_classifier.validation_step(
            (features.detach(), targets), batch_idx
        )
        self.log_dict(cls_log, 
                      prog_bar=True,
                      sync_dist=True, 
                      batch_size=len(targets))

        self.log_dict({"val_e_rank":effective_rank(features.to(torch.float32))[0]}, 
                    batch_size=len(targets),
                    sync_dist=True)

        return cls_loss

    def configure_optimizers(self):
        params_weight_decay, params_no_weight_decay = get_weight_decay_parameters(
                    [self.backbone, self.projection_head, self.student_head]                    
                )
        optimizer = SGD(
            [
                {   "name": "params_weight_decay", 
                    "params": params_weight_decay,
                    "weight_decay": self.decay,
                },
                {   "name": "params_no_weight_decay", 
                    "params": params_no_weight_decay,                    
                },                
                {   "name": "online_classifier",
                    "params": self.online_classifier.parameters(),
                    # "lr": self.linear_lr * self.batch_size_per_device * self.trainer.world_size / 256,
                },
                {   "name": "var_regressor",
                    "params": self.var_head.parameters(),                    
                }               
            ],            
            lr=self.lr * self.batch_size_per_device * self.trainer.world_size / 256,
            momentum=0.9,
            weight_decay=0.0,
        )         
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=int(
                      self.trainer.estimated_stepping_batches
                    / self.trainer.max_epochs
                    * self.warmup
                ),                
                max_epochs=int(self.trainer.estimated_stepping_batches),
            ),
            "interval": "step",
        }

        return [optimizer], [scheduler]

train_identity= lambda NORMALIZE: T.Compose([                    
                    T.RandomHorizontalFlip(), T.ToImage(),  T.ToDtype(torch.float32, scale=True),
                    T.Normalize(mean=NORMALIZE["mean"], std=NORMALIZE["std"]),
                ])
# For ResNet50 we adjust crop scales as recommended by the authors:
# https://github.com/facebookresearch/dino#resnet-50-and-other-convnets-trainings
# transform = DINOTransform(global_crop_scale=(0.14, 1), local_crop_scale=(0.05, 0.14), n_local_views=n_local_views)
def train_transform(size, scale=(0.08, 1.0), NORMALIZE=IMAGENET_NORMALIZE):
    return T.Compose([
                    T.RandomResizedCrop(size, scale=scale),
                    T.RandomHorizontalFlip(), T.ToImage(),  T.ToDtype(torch.float32, scale=True),
                    T.Normalize(mean=NORMALIZE["mean"], std=NORMALIZE["std"]),
                ])
val_identity  = lambda size, NORMALIZE: T.Compose([
                    T.Resize(size), T.ToImage(),  T.ToDtype(torch.float32, scale=True),
                    T.Normalize(mean=NORMALIZE["mean"], std=NORMALIZE["std"]),
                ])
val_transform = T.Compose([
                    T.Resize(256), T.CenterCrop(224), T.ToImage(),  T.ToDtype(torch.float32, scale=True),
                    T.Normalize(mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]),
                ])

CIFAR10_NORMALIZE   = {'mean':(0.4914, 0.4822, 0.4465), 'std':(0.2470, 0.2435, 0.2616)}
CIFAR100_NORMALIZE  = {'mean':(0.5071, 0.4867, 0.4408), 'std':(0.2675, 0.2565, 0.2761)}
TINYIMAGE_NORMALIZE = {'mean':(0.4802, 0.4481, 0.3975), 'std':(0.2302, 0.2265, 0.2262)}
STL10_NORMALIZE     = {'mean':(0.4408, 0.4279, 0.3867), 'std':(0.2682, 0.2610, 0.2686)}

transforms = {
"Cifar10-2":    DINOTransform(global_crop_size=32,
                            global_crop_scale=(0.08, 1.0),
                            n_local_views=0,
                            gaussian_blur=(0.5, 0.0, 0.0),
                            normalize=CIFAR10_NORMALIZE),

"Cifar100-2":   DINOTransform(global_crop_size=32,
                            global_crop_scale=(0.08, 1.0),
                            n_local_views=0,
                            gaussian_blur=(0.5, 0.0, 0.0),
                            normalize=CIFAR100_NORMALIZE),

"Tiny-2":       DINOTransform(global_crop_size=64,
                            global_crop_scale=(0.08, 1.0),
                            n_local_views=0,
                            gaussian_blur=(0.5, 0.0, 0.0),
                            normalize=TINYIMAGE_NORMALIZE),

"STL-2":        DINOTransform(global_crop_size=96,
                            global_crop_scale=(0.20, 1.0),
                            n_local_views=0,
                            gaussian_blur=(0.5, 0.0, 0.0),
                            normalize=STL10_NORMALIZE),

"Im100-2":      DINOTransform(global_crop_scale=(0.08, 1.0),
                            n_local_views=0),

"Im1k-8":       DINOTransform(global_crop_scale=(0.08, 1.00),
                            local_crop_scale =(0.05, 0.14)),
"Im1k-2":       DINOTransform(global_crop_scale=(0.08, 1.00),
                            n_local_views=0),
}

train_transforms = {
"Cifar10":   train_identity(CIFAR10_NORMALIZE),
"Cifar100":  train_identity(CIFAR100_NORMALIZE),
"Tiny":      train_identity(TINYIMAGE_NORMALIZE),
"STL":       train_transform(96, NORMALIZE=STL10_NORMALIZE),
"Im100":     train_transform(224),
"Im1k":      train_transform(224),
}

val_transforms = {
"Cifar10":   val_identity(32, CIFAR10_NORMALIZE),
"Cifar100":  val_identity(32, CIFAR100_NORMALIZE),
"Tiny":      val_identity(64, TINYIMAGE_NORMALIZE),
"STL":       val_identity(96, STL10_NORMALIZE),
"Im100":     val_transform,
"Im1k":      val_transform,
}



