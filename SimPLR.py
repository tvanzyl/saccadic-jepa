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


class SimPLR(LightningModule):
    def __init__(self, batch_size_per_device: int, 
                 num_classes: int, 
                 resnetsize:int = 50,
                 n_local_views:int = 6,
                 lr:float = 0.15,
                 decay:float=1e-4) -> None:
        super().__init__()        
        self.save_hyperparameters('batch_size_per_device',
                                  'num_classes',
                                  'resnetsize',
                                  'n_local_views',
                                  'lr' )

        self.lr = lr
        self.decay = decay
        self.batch_size_per_device = batch_size_per_device

        if resnetsize == 18:
            resnet = resnet18()
        elif resnetsize == 34:
            resnet = resnet34()
        else:
            resnet = resnet50()
        
        self.emb_width = emb_width = resnet.fc.in_features
        resnet.fc = Identity()  # Ignore classification head
        
        upd_width = 2048
        prd_width = 256
        self.ens_size = 2 + n_local_views

        self.backbone = resnet

        self.projection_head = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(emb_width, upd_width)),
                # nn.BatchNorm1d(upd_width),
                nn.GELU(),
                # nn.Linear(upd_width, upd_width, True),
                L2NormalizationLayer(),
                nn.utils.weight_norm(nn.Linear(upd_width, emb_width)),
                nn.BatchNorm1d(emb_width, affine=False),
                nn.GELU(),
            )                
        self.prediction_head = nn.Sequential(
            nn.Linear(emb_width, prd_width, False),
            # nn.LeakyReLU()
        )
        self.merge_head = nn.Sequential(
            nn.Linear(emb_width, prd_width),
            # nn.LeakyReLU()
        )
        self.merge_head[0].weight.data = self.prediction_head[0].weight.data.clone()
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
            [self.backbone] #, self.projection_head, self.prediction_head]
        )
        #DIET AdamW 0.001/0.05, warmup 10
        # optimizer = AdamW(
        optimizer = SGD(        
            [
                {"name": "simplr", "params": params},
                {
                    "name": "proj", 
                    "params": self.projection_head.parameters(),
                },
                {
                    "name": "pred", 
                    "params": self.prediction_head.parameters(),
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

    # def configure_gradient_clipping(
    #     self,
    #     optimizer: Optimizer,
    #     gradient_clip_val: Union[int, float, None] = None,
    #     gradient_clip_algorithm: Union[str, None] = None,
    # ) -> None:
    #     self.clip_gradients(
    #         optimizer=optimizer,
    #         gradient_clip_val=3.0,
    #         gradient_clip_algorithm="norm",
    #     )
    #     self.student_projection_head.cancel_last_layer_gradients(self.current_epoch)

# For ResNet50 we adjust crop scales as recommended by the authors:
# https://github.com/facebookresearch/dino#resnet-50-and-other-convnets-trainings
# transform = DINOTransform(global_crop_scale=(0.14, 1), local_crop_scale=(0.05, 0.14), n_local_views=n_local_views)
transform = DINOTransform(global_crop_scale=(0.2, 1), local_crop_scale=(0.05, 0.2))
transforms = {
"Cifar10": transform,
"Cifar100":transform,
"Tiny":    transform,
"Nette":   transform,
"Im100":   transform,
"Im1k":    transform,
"Im100-2": DINOTransform(global_crop_scale=(0.2, 1), 
                         n_local_views=2,
                         local_crop_scale=(0.05, 0.2)),
"Im1k-2":  DINOTransform(global_crop_scale=(0.2, 1), 
                         n_local_views=2,
                         local_crop_scale=(0.05, 0.2)),
}

val_transform = T.Compose([
                    T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize(mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]),
                ])

val_transforms = {
"Cifar10": val_transform,
"Cifar100":val_transform,
"Tiny":    val_transform,
"Nette":   val_transform,
"Im100":   val_transform,
"Im1k":    val_transform,
"Im100-2": val_transform,
"Im1k-2":  val_transform,
}
