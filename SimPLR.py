import math
import copy
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import SGD, AdamW

from torchvision.models import resnet50, resnet34 , resnet18
from torchvision.transforms import v2 as T

from pytorch_lightning import LightningModule

from lightly.transforms import DINOTransform
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.loss import NegativeCosineSimilarity
from lightly.models.utils import (
    get_weight_decay_parameters,
    deactivate_requires_grad, 
    update_momentum,
    update_drop_path_rate
)
# from lightly.models._momentum import _do_momentum_update

from lightly.utils.benchmarking import OnlineLinearClassifier
from lightly.utils.scheduler import CosineWarmupScheduler, cosine_schedule
from lightly.utils.debug import std_of_l2_normalized

from timm.models.vision_transformer import vit_small_patch16_224
from lightly.models.modules import MaskedVisionTransformerTIMM

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
    GEMINI GENERATED: Computes the effective rank and condition number of a batch of embeddings.
    
    Args:
        embeddings: Tensor of shape [Batch_Size, Dimension]
        eps: Small value for numerical stability
        
    Returns:
        effective_rank: Float representing the continuous dimensionality.
        condition_number: Float representing the ratio of max to min variance.
    """
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    _, S, _ = torch.linalg.svd(centered, full_matrices=False)
    eigenvalues = (S ** 2) / (centered.size(0) - 1)
    p = eigenvalues / (eigenvalues.sum() + eps)
    entropy = -torch.sum(p * torch.log(p + eps))
    effective_rank = torch.exp(entropy).item()
    condition_number = (eigenvalues[0] / (eigenvalues[-1] + eps)).item()    
    return effective_rank, condition_number


def backbones(name):
    if name in ["resnetjie-9","resnetjie-18"]:
        resnet = {"resnetjie-9" :resnet18,
                  "resnetjie-18":resnet18}[name]()
        emb_width = resnet.fc.in_features
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
        resnet.maxpool = nn.Sequential()
        resnet.fc = nn.Identity()
        backbone = resnet
    elif name in ["resnet-18", "resnet-34", "resnet-50"]: 
        resnet = {"resnet-18":resnet18, 
                  "resnet-34":resnet34, 
                  "resnet-50":resnet50}[name](zero_init_residual=(name!="resnet-18"))
        emb_width = resnet.fc.in_features
        resnet.fc = nn.Identity()
        backbone = resnet
    elif name in ["vit-s/16"]:
        vit = vit_small_patch16_224(dynamic_img_size=True)
        mvt = MaskedVisionTransformerTIMM(vit=vit)
        emb_width = mvt.vit.embed_dim
        backbone = mvt
    else:
        raise NotImplemented("Backbone Not Supported")
    print(f"Emb Width {emb_width}")
    return backbone, emb_width

class SimPLR(LightningModule):
    def __init__(self, batch_size_per_device: int,
                 num_classes: int, 
                 warmup: int = 0,
                 backbone:str = "resnet-50",
                 lr:float = 1.0, 
                 decay:float=1e-5,
                 random_head:bool=False,
                 no_projection_head:bool=False,
                 alpha:float = 0.9, 
                 cut:float = 0.0,
                 prd_width:int = 256,
                 prj_depth:int = 2,
                 prj_width:int = 2048,
                 no_buttress:bool=False,
                 no_student_head:bool=False,
                 JS:bool=False, 
                 ema:bool=False, 
                 var:float=0.0,
                 AdamW:bool=False,
                 accumulate:int=1,
                 ) -> None:
        super().__init__()
        self.save_hyperparameters('batch_size_per_device',
                                  'num_classes', 'warmup',
                                  'backbone',
                                  'lr', 'decay', 'JS',
                                  'random_head',
                                  'no_projection_head',
                                  'no_student_head',
                                  'alpha', 
                                  'cut','prd_width', 
                                  "prj_depth", "prj_width",
                                  'no_buttress',
                                  'ema', 'var', 'AdamW')

        self.warmup = warmup
        self.lr = lr        
        self.decay = decay
        self.batch_size_per_device = batch_size_per_device
        self.JS = JS        
        self.ema = ema
        self.var = torch.tensor(var, device=self.device, requires_grad=False)
        self.alpha = alpha
        self.prd_width = prd_width
        self.AdamW = AdamW
        self.accumulate = accumulate
                
        identity_head = not random_head
        
        self.backbone, self.emb_width = backbones(backbone)
        emb_width = self.emb_width

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
                                         nn.Linear(prj_width, prj_width, bias=(i==prj_depth-1))])
        
        if self.ema:
            self.teacher_backbone = copy.deepcopy(self.backbone)
            deactivate_requires_grad(self.teacher_backbone)
            self.teacher_projection_head = copy.deepcopy(self.projection_head)
            deactivate_requires_grad(self.teacher_projection_head)
        
        #changed after the copy.
        if backbone in ["vit-s/16"]:
            update_drop_path_rate(
                self.backbone.vit,
                drop_path_rate=0.1,  # we recommend using smaller rates like 0.1 for vit-s-14
                mode="uniform",
            )
            if self.ema:
                self.teacher_backbone.eval()

        #Use Batchnorm non-affine for centering
        if no_buttress:
            self.buttress = nn.Identity()
        else:
            self.buttress = nn.BatchNorm1d(prj_width, affine=False)
            
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
        self.teacher_head = nn.Sequential(nn.ReLU(), teacher_head)
        deactivate_requires_grad(self.teacher_head)

        if no_student_head:
            student_head = nn.AdaptiveAvgPool1d(prd_width)
        else:
            student_head = nn.Linear(prj_width, prd_width, False)
            if random_head:
                if cut == 0.0:
                    cut = (teacher_head.weight.data.var()/student_head.weight.data.var())**0.5
                    print(f"Cut: {cut}")
                student_head.weight.data = teacher_head.weight.data.clone()
            if cut > 0.0: # https://arxiv.org/pdf/2406.16468 (Cut Init)
                student_head.weight.data.div_(cut)
        self.student_head = nn.Sequential(nn.ReLU(), student_head)
        
        self.online_classifier = OnlineLinearClassifier(feature_dim=emb_width, num_classes=num_classes)

        self.criterion = NegativeCosineSimilarity()

    def forward(self, x: Tensor) -> Tensor:
        if self.ema:
            return self.teacher_backbone(x)
        else:
            return self.backbone(x)
    
    def forward_student(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        h = self.backbone( x ).flatten(start_dim=1)
        z = self.projection_head( h )
        p = self.student_head( z )
        return p, h.detach(), z.detach()
    
    def forward_teacher(self, x: Tensor, z: Tensor) -> Tensor:
        if self.ema:
            h = self.teacher_backbone( x ).flatten(start_dim=1)
            z = self.teacher_projection_head( h ).detach()
        b = self.buttress( z )
        q = self.teacher_head( b )
        return q    

    def forward_JS(self, x: List[Tensor], idx: Tensor) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        # Two globals
        p0_, h0_, z0_ = self.forward_student(x[0])
        p1_, h1_, z1_ = self.forward_student(x[1])
        
        q0_ = self.forward_teacher(x[0], z0_)
        q1_ = self.forward_teacher(x[1], z1_)

        if self.JS: # For James-Stein
            #EMM, EWM-A/V https://fanf2.user.srcf.net/hermes/doc/antiforgery/stats.pdf
            mean_ = self.embedding[idx]

            qdiff0_ = q0_  - mean_
            qdiff1_ = q1_  - mean_

            if self.var > 0.0:
                var_ = self.var 
            else:
                var_ = torch.mean( qdiff0_.abs()*qdiff1_.abs() )

            # https://openaccess.thecvf.com/content/WACV2024/papers/Khoshsirat_Improving_Normalization_With_the_James-Stein_Estimator_WACV_2024_paper.pdf
            norm0_ = torch.linalg.vector_norm((qdiff0_)/((var_**0.5)+1e-9), dim=1, keepdim=True)**2
            norm1_ = torch.linalg.vector_norm((qdiff1_)/((var_**0.5)+1e-9), dim=1, keepdim=True)**2

            n0 = torch.maximum(1.0 - (self.prd_width-2.0)/(norm0_+1e-9), torch.tensor(0.0))
            n1 = torch.maximum(1.0 - (self.prd_width-2.0)/(norm1_+1e-9), torch.tensor(0.0))

            q0_ = n0*q0_ + (1.-n0)*mean_
            q1_ = n1*q1_ + (1.-n1)*mean_
            
            alpha = cosine_schedule(self.global_step, self.trainer.estimated_stepping_batches, 
                                    1.000, self.alpha)
            incr_ = alpha*(qdiff0_+ qdiff1_)/2.0
            self.embedding[idx] = (mean_ + incr_).to(torch.float32)

            self.log_dict({"JS_n0_n1":n0.mean()})
            self.log_dict({"var":torch.mean(var_)})

        p = [p0_, p1_]
        q = [q1_, q0_]

        if len(x) > 2: # MultiCrops            
            q_ = 0.5*(q0_+q1_)
            p.extend([self.forward_student(x_)[0] for x_ in x[2:]])
            q.extend([q_ for _ in range(len(x)-2)])
        
        return h0_, p, q

    def on_train_start(self):
        if self.JS:
            N = len(self.trainer.train_dataloader.dataset)
            embedding  = torch.zeros((N, self.prd_width),                                        
                                        device=self.device,
                                        requires_grad=False)
            self.register_buffer("embedding", embedding)
        return super().on_train_start()

    def training_step(
        self, batch: Tuple[List[Tensor], Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        x, targets, idx = batch
        
        if self.ema: #These lines give us classical EMA 
            momentum = cosine_schedule(self.global_step, self.trainer.estimated_stepping_batches, 
                                       0.996, 1)
            update_momentum(self.backbone, self.teacher_backbone, m=momentum)            
            update_momentum(self.projection_head, self.teacher_projection_head, m=momentum)
        
        h0_, p, q = self.forward_JS( x, idx )

        loss = 0
        for xi in range(len(q)):
            loss += self.criterion( p[xi], q[xi] ) / len(q)
        
        self.log_dict(
            {"train_loss": loss},
            prog_bar=True,
            sync_dist=True,
            batch_size=len(targets),
        )

        # Online classification.
        cls_loss, cls_log = self.online_classifier.training_step(
            (h0_, targets), batch_idx
        )

        self.log_dict(
            cls_log, 
            sync_dist=True, 
            batch_size=len(targets))

        return loss + cls_loss #+ var_loss

    def validation_step(
        self, batch: Tuple[Tensor, Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        images, targets = batch[0], batch[1]
        features = self.forward(images).flatten(start_dim=1)
        cls_loss, cls_log = self.online_classifier.validation_step(
            (features, targets), batch_idx
        )
        self.log_dict(cls_log, 
                      prog_bar=True,
                      sync_dist=True, 
                      batch_size=len(targets))

        z = self.projection_head( features )
        p = self.student_head( z )
        b = self.buttress( z )
        q = self.teacher_head( b )

        self.log_dict(            
            {"h_quality":std_of_l2_normalized(features)},
            batch_size=len(targets),
            sync_dist=True)
        self.log_dict(
            {"butt_e_rank":effective_rank(z.to(torch.float32))[0]},
            batch_size=len(targets),
            sync_dist=True)
        self.log_dict(
            {"student_e_rank":effective_rank(p.to(torch.float32))[0]},
            batch_size=len(targets),
            sync_dist=True)
        self.log_dict(
            {"teacher_e_rank":effective_rank(q.to(torch.float32))[0]},
            batch_size=len(targets),
            sync_dist=True)
        self.log_dict(
            {"val_e_rank":effective_rank(features.to(torch.float32))[0]}, 
            batch_size=len(targets),
            sync_dist=True)

        return cls_loss

    def configure_optimizers(self):
        params_weight_decay, params_no_weight_decay = get_weight_decay_parameters(
                    [self.backbone, self.projection_head, self.student_head]                    
                )
        param_cfg = [
                        {   "name": "params_weight_decay", 
                            "params": params_weight_decay,
                            "weight_decay": self.decay,
                        },
                        {   "name": "params_no_weight_decay", 
                            "params": params_no_weight_decay,                    
                        },                
                        {   "name": "online_classifier",
                            "params": self.online_classifier.parameters(),
                        },
                    ]
        if self.AdamW:      
            optimizer = AdamW(param_cfg,
                lr=self.lr * self.accumulate * self.batch_size_per_device * self.trainer.world_size / 256,
                weight_decay=0.0,
            )
        else:  
            optimizer = SGD(param_cfg,
                lr=self.lr * self.accumulate * self.batch_size_per_device * self.trainer.world_size / 256,
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
                            global_crop_scale=(0.08, 1.0),
                            n_local_views=0,
                            gaussian_blur=(0.5, 0.0, 0.0),
                            normalize=STL10_NORMALIZE),

"Im100-2":      DINOTransform(global_crop_scale=(0.08, 1.0),
                            n_local_views=0),
"Im100-8":      DINOTransform(),

"Im1k-2":       DINOTransform(global_crop_scale=(0.08, 1.00),
                            n_local_views=0),
"Im1k-8":       DINOTransform(),
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



