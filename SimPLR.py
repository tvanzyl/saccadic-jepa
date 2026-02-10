import math
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
from torch.nn import Identity
from torch.optim import SGD

from torchvision import transforms as T
from torchvision.models import resnet50, resnet34 , resnet18
# from resnet import resnet18, resnet34, resnet50

from pytorch_lightning import LightningModule

from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.models._momentum import _do_momentum_update
from lightly.loss import NegativeCosineSimilarity
from lightly.models.utils import get_weight_decay_parameters

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

class BiasLayer(nn.Module):
    def __init__(self, size:int):
        super(BiasLayer, self).__init__()
        self.bias = nn.Parameter(torch.zeros(size))
        bound_w = 1 / math.sqrt(size)
        nn.init.normal_(self.bias, 0, bound_w)

    def forward(self, x):
        return x + self.bias

def update_bn_params(model, momentum=0.01, eps=1e-5):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum
            m.eps = eps

def backbones(name):
    if name in ["resnetjie-9","resnetjie-18"]:
        resnet = {"resnetjie-9" :resnet18,
                  "resnetjie-18":resnet18}[name]()
        emb_width = resnet.fc.in_features
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
        resnet.maxpool = nn.Sequential()
        resnet.fc = Identity()
    elif name in ["resnet-18", "resnet-34", "resnet-50"]: 
        resnet = {"resnet-18":resnet18, 
                  "resnet-34":resnet34, 
                  "resnet-50":resnet50}[name](zero_init_residual=(name!="resnet-18"))
        emb_width = resnet.fc.in_features
        resnet.fc = Identity()
        # update_bn_params(resnet, momentum=0.01)
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
                 alpha:float = 1.00, gamma:float = 0.50,
                 cut:float = 2.0,
                 prd_width:int = 256,
                 prj_depth:int = 2,
                 prj_width:int = 2048,
                 L2:bool=False,
                 no_ReLU_buttress:bool=False,
                 no_student_head:bool=False,
                 JS:bool=False, 
                 no_bias:bool=False,
                 emm:bool=False, emm_v:int=0, var:float=0.1,
                 fwd:int=0,
                 end_value:float=0.001) -> None:
        super().__init__()
        self.save_hyperparameters('batch_size_per_device',
                                  'num_classes', 'warmup',
                                  'backbone',
                                  'n_local_views',
                                  'lr', 'decay', 'JS',
                                  'momentum_head',
                                  'identity_head',
                                  'no_projection_head',
                                  'no_student_head',
                                  'alpha', 'gamma',
                                  'cut','prd_width', 
                                  "prj_depth", "prj_width", 'L2',
                                  'no_ReLU_buttress',
                                  'emm', 'emm_v', 'var',
                                  'no_bias',
                                  'fwd', 
                                  'end_value')
        self.warmup = warmup
        self.lr = lr
        self.decay = decay
        self.batch_size_per_device = batch_size_per_device                
        self.JS = JS
        self.emm = emm
        self.emm_v = emm_v
        self.var = var
        self.fwd = fwd        
        self.momentum_head = momentum_head
        self.alpha = alpha
        self.gamma = gamma
        self.no_ReLU_buttress = no_ReLU_buttress
        self.end_value = end_value

        if identity_head and momentum_head:
            raise Exception("Invalid Arguments, can't select identity and momentum")
        if JS and not emm and fwd == 0:
            raise Exception("Invalid Arguments, Need One of Fwd or EMM with JS")        
        
        resnet, emb_width = backbones(backbone)
        self.emb_width  = emb_width # Used by eval classes        
        
        self.backbone = resnet

        if no_projection_head:
            prj_width = self.emb_width
        
        self.prd_width = prd_width
        
        self.projection_head = nn.Sequential()
        if L2:
            self.projection_head.extend([L2NormalizationLayer(),])
        if not no_projection_head:
            self.projection_head.extend([nn.Linear(emb_width, prj_width, bias=False),])
            for i in range(prj_depth):
                self.projection_head.extend(
                                    [nn.BatchNorm1d(prj_width),
                                     nn.ReLU(),
                                     nn.Linear(prj_width, prj_width, bias=(i<prj_depth-1)),]
                )

        #Use Batchnorm none-affine for centering
        self.buttress = nn.BatchNorm1d(prj_width, 
                                       affine=False)

        if identity_head:
            if prj_width == prd_width:
                teacher_head = nn.Identity()
            elif prj_width > prd_width:
                #Identity matrix hack for if requires dimensionality reduction
                teacher_head = nn.AdaptiveAvgPool1d(self.prd_width)
            else:
                raise NotImplementedError("Invalid Arguments, can't select prd width larger than prj width")
        else:            
            teacher_head = nn.Linear(prj_width, self.prd_width, False)
            nn.init.orthogonal_(teacher_head.weight)
        self.teacher_head = nn.Sequential(teacher_head)

        if no_student_head:
            student_head = nn.AdaptiveAvgPool1d(self.prd_width)
        else:
            student_head = nn.Linear(prj_width, self.prd_width, False)
            if cut == 0.0:
                cut = (teacher_head.weight.data.var()/student_head.weight.data.var())**0.5
                print(f"Cut: {cut}")
            student_head.weight.data = teacher_head.weight.data.clone()
            if cut > 0.0: # https://arxiv.org/pdf/2406.16468 (Cut Init)
                student_head.weight.data.div_(cut)
        self.student_head = nn.Sequential(student_head)        

        if not no_bias:
            biaslayer = BiasLayer(prj_width)
            if cut > 0.0: # https://arxiv.org/pdf/2406.16468 (Cut Init)
                biaslayer.bias.data.div_(cut)
            self.teacher_head.insert(0, biaslayer)

        if not no_ReLU_buttress:
            self.student_head.insert(0, nn.ReLU())
            self.teacher_head.insert(0, nn.ReLU())

        self.var_head = nn.Sequential(                                     
                                nn.Linear(prj_width, prj_width),
                                nn.ReLU(),
                                nn.Linear(prj_width, prd_width),
                                nn.Softplus()
                            )
        
        self.online_classifier = OnlineLinearClassifier(feature_dim=emb_width, num_classes=num_classes)
        
        self.criterion = NegativeCosineSimilarity()
        self.var_crt = nn.GaussianNLLLoss()

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def forward_student(self, x: List[Tensor], idx: Tensor) -> Tensor:
        views = len(x)

        # Two globals
        h = [self.backbone( x_ ).flatten(start_dim=1) for x_ in x[:2]]
        h0_ = h[0].detach()        
        z = [self.projection_head( h_ ) for h_ in h]
        p = [self.student_head( z_ ) for z_ in z]
        
        if views > self.fwd + 2: # MultiCrops
            h_multi = [self.backbone( x_ ).flatten(start_dim=1) for x_ in x[self.fwd+2:]]
            z_multi = [self.projection_head( h_ ) for h_ in h_multi]
            p.extend([self.student_head( z_ ) for z_ in z_multi])        

        vars = [self.var_head( z_.detach() ) for z_ in z]
        
        with torch.no_grad():
            self.log_dict({"h_quality":std_of_l2_normalized(h0_)})

            b = [self.buttress( z_.detach() ) for z_ in z]
            qo = [self.teacher_head( b_ ) for b_ in b]
            q0_ = qo[0]
            q1_ = qo[1]

            # Fwds Only
            if self.fwd > 0:
                # self.train(False)
                h_fwd = [self.backbone( x_ ).flatten(start_dim=1) for x_ in x[2:self.fwd+2]]
                z_fwd = [self.projection_head( h_ ) for h_ in h_fwd]                
                b_fwd = [self.buttress( z_ ) for z_ in z_fwd]
                q_fwd = [self.teacher_head( b_ ) for b_ in b_fwd]
                # q_fwd = [self.student_head( z_ ) for z_ in z_fwd]
                # q_fwd.extend(p_fwd)
                # self.train(True)

            if self.JS: # For James-Stein                                                 
                if self.current_epoch == 0 and self.emm:
                    self.embedding[idx] = 0.5*(q0_+q1_)
                else:
                    if self.fwd: #Use forwards as the mean
                        mean_ = torch.mean(torch.stack(q_fwd, dim=0), dim=0) #speed
                    elif self.emm: #EMM 
                        # EWM-A/V https://fanf2.user.srcf.net/hermes/doc/antiforgery/stats.pdf
                        mean_ = self.embedding[idx]

                    qdiff0_ = q0_  - mean_
                    qdiff1_ = q1_  - mean_
                    
                    if self.emm_v == 9:
                        var_ = torch.tensor(self.var, device=self.device)
                    elif self.emm_v == 8:
                        var_ = torch.mean(torch.stack(vars, dim=0), dim=0).detach()
                    elif self.emm_v == 5:
                        qmean_ = torch.mean(torch.stack(qo, dim=0), dim=0)
                        qdiff_s = [(q_fwd_ - qmean_)**2 for q_fwd_ in q_fwd]
                        var_ =  self.gamma*torch.mean(torch.stack(qdiff_s, dim=0), dim=0)
                    else:
                        raise Exception("Not Valid EMM V")

                    # https://openaccess.thecvf.com/content/WACV2024/papers/Khoshsirat_Improving_Normalization_With_the_James-Stein_Estimator_WACV_2024_paper.pdf
                    norm0_ = torch.linalg.vector_norm((qdiff0_)/((var_**0.5)+1e-9), dim=1, keepdim=True)**2
                    norm1_ = torch.linalg.vector_norm((qdiff1_)/((var_**0.5)+1e-9), dim=1, keepdim=True)**2

                    n0 = torch.maximum(1.0 - (self.prd_width-2.0)/(norm0_+1e-9), torch.tensor(0.0))
                    n1 = torch.maximum(1.0 - (self.prd_width-2.0)/(norm1_+1e-9), torch.tensor(0.0))

                    self.log_dict({"qdiff":qdiff0_.mean()})
                    self.log_dict({"JS_n0_n1":n0.mean()})
                    self.log_dict({"var":torch.mean(var_)})
                    self.log_dict({"var_var":torch.var(var_)})

                    q0_ = n0*q0_ + (1.-n0)*mean_
                    q1_ = n1*q1_ + (1.-n1)*mean_
                    
                    if self.emm:
                        # zic_ = (qincr0_+qincr1_)/2.0
                        self.embedding[idx] = mean_ + self.alpha*(qdiff0_+ qdiff1_)/2.0
                    elif self.fwd:
                        pass
                    else:
                        raise Exception("Not Valid Combo")
            
            q  = [q1_, q0_]            
            if views > self.fwd + 2:
                q_ = 0.5*(q0_+q1_)
                q.extend([q_ for _ in range(views-2-self.fwd)])            
            assert len(p)==len(q)
        return h0_, p, q, qo, vars

    def on_train_start(self):
        if self.JS:
            N = len(self.trainer.train_dataloader.dataset)
            if self.emm:
                self.embedding      = torch.empty((N, self.prd_width),
                                            dtype=torch.float16,
                                            device=self.device)            
        return super().on_train_start()

    def training_step(
        self, batch: Tuple[List[Tensor], Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        x, targets, idx = batch
        
        h0_, p, q, qo, vars = self.forward_student( x, idx )

        loss = 0
        var_loss = 0        
        qomean_ = torch.mean(torch.stack(qo, dim=0), dim=0).detach()
        for xi in range(len(q)):
            p_ = p[xi]
            q_ = q[xi]
            loss += self.criterion( p_, q_ ) / len(q)

            var_  = vars[xi]
            qo_    = qo[xi].detach()
            var_loss += self.var_crt(math.sqrt(2)*qomean_, math.sqrt(2)*qo_, var_)

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

        #These lines give us classical EMA v1
        if self.momentum_head:
            momentum = cosine_schedule(self.global_step, self.trainer.estimated_stepping_batches, 0.996, 1)
            _do_momentum_update(self.teacher_head.weight, self.student_head.weight, momentum)

        return loss + cls_loss + var_loss

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
                    [self.backbone, self.projection_head, self.student_head]                    
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
                {   "name": "var_regressor",
                    "params": self.var_head.parameters(),
                    "weight_decay": 0.0,
                }               
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
"Cifar10-2":    DINOTransform(global_crop_size=32,
                            global_crop_scale=(0.08, 1.0),
                            n_local_views=0,
                            gaussian_blur=(0.5, 0.0, 0.0),
                            normalize=CIFAR10_NORMALIZE),
"Cifar10-4":    DINOTransform(global_crop_size=32,
                            global_crop_scale=(0.08, 1.0),
                            n_local_views=2,
                            local_crop_size=32,
                            local_crop_scale=(0.08, 1.0),
                            gaussian_blur=(0.5, 0.0, 0.0),
                            normalize=CIFAR10_NORMALIZE),

"Cifar100-2":   DINOTransform(global_crop_size=32,
                            global_crop_scale=(0.08, 1.0),
                            n_local_views=0,
                            gaussian_blur=(0.5, 0.0, 0.0),
                            normalize=CIFAR100_NORMALIZE),
"Cifar100-4":   DINOTransform(global_crop_size=32,
                            global_crop_scale=(0.08, 1.0),
                            n_local_views=2,
                            local_crop_size=32,
                            local_crop_scale=(0.08, 1.0),
                            gaussian_blur=(0.5, 0.0, 0.0),
                            normalize=CIFAR100_NORMALIZE),

"Tiny-2":       DINOTransform(global_crop_size=64,
                            global_crop_scale=(0.14, 1.0),
                            n_local_views=0,
                            gaussian_blur=(0.5, 0.0, 0.0),
                            normalize=TINYIMAGE_NORMALIZE),
"Tiny-4":       DINOTransform(global_crop_size=64,
                            global_crop_scale=(0.08, 1.0),
                            n_local_views=2,
                            local_crop_size=64,
                            local_crop_scale=(0.08, 1.0),
                            gaussian_blur=(0.5, 0.0, 0.0),
                            normalize=TINYIMAGE_NORMALIZE),

"STL-2":        DINOTransform(global_crop_size=96,
                            global_crop_scale=(0.20, 1.0),
                            n_local_views=0,
                            gaussian_blur=(0.5, 0.0, 0.0),
                            normalize=STL10_NORMALIZE),

"Im100-6":      JSREPATransform(global_crop_scale=(0.08, 1.0),
                            n_global_views=6),
"Im100-4":      JSREPATransform(global_crop_scale=(0.08, 1.0),
                            n_global_views=4),
"Im100-4-08":   DINOTransform(global_crop_scale=(0.08, 1.0),
                            n_local_views=2,
                            local_crop_size=224,
                            local_crop_scale=(0.08, 1.0),),
"Im100-4-14":   DINOTransform(global_crop_scale=(0.08, 1.0),
                            n_local_views=2,
                            local_crop_size=224,
                            local_crop_scale=(0.14, 1.0),),
"Im100-2-20":   DINOTransform(global_crop_scale=(0.20, 1.0),
                            n_local_views=0),
"Im100-2-14":   DINOTransform(global_crop_scale=(0.14, 1.0),
                            n_local_views=0),
"Im100-2-08":   DINOTransform(global_crop_scale=(0.08, 1.0),
                            n_local_views=0),
"Im100-2-05":   DINOTransform(global_crop_scale=(0.05, 1.0),
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


