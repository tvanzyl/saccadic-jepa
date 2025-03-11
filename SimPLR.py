import copy
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Identity
from torch.optim import SGD
from torch.optim.optimizer import Optimizer
from torchvision.models import resnet50, resnet34, resnet18

from lightly.loss import NegativeCosineSimilarity
from lightly.models.utils import (
    get_weight_decay_parameters,
)

from lightly.transforms import DINOTransform
from lightly.utils.benchmarking import OnlineLinearClassifier
from lightly.utils.scheduler import CosineWarmupScheduler, cosine_schedule

n_local_views = 0

class L2NormalizationLayer(nn.Module):
    def __init__(self, dim=1, eps=1e-12):
        super(L2NormalizationLayer, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:    
        return nn.normalize(x, p=2, dim=self.dim, eps=self.eps)


class SimPLR(LightningModule):
    def __init__(self, batch_size_per_device: int, 
                 num_classes: int, 
                 resnetsize:int = 50,                 
                 n_local_views:int = n_local_views,
                 lr:float = 0.15) -> None:
        super().__init__()        
        self.save_hyperparameters('batch_size_per_device',
                                  'num_classes', 
                                  'resnetsize',                                 
                                  'n_local_views',
                                  'lr' )

        self.lr = lr        
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
                nn.Linear(emb_width, upd_width, affine=False),
                nn.BatchNorm1d(upd_width),
                nn.ReLU(inplace=True),
                nn.Linear(upd_width, emb_width),
                L2NormalizationLayer(),
                nn.BatchNorm1d(emb_width, affine=False),
                nn.LeakyReLU(),
            )        
        self.prediction_head = nn.Linear(emb_width, prd_width, False)
        self.merge_head = nn.Linear(emb_width, prd_width)
        self.merge_head.weight.data = self.prediction_head.weight.data
        self.criterion = NegativeCosineSimilarity()

        self.online_classifier = OnlineLinearClassifier(feature_dim=emb_width, num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def forward_student_opt(self, x: Tensor) -> Tensor:
        fg = self.backbone( torch.cat( x[:2] ) ).flatten(start_dim=1)
        f0_ = fg[self.batch_size_per_device].detach()
        gg = self.projection_head( fg )
        pg = self.prediction_head( gg )
        p = torch.chunk( pg, self.ens_size )
        if self.ens_size > 2:
            fl = self.backbone( torch.cat( x[2:] ) ).flatten(start_dim=1)
            gl = self.projection_head( fl )
            pl = self.prediction_head( gl )
            p.append( pl )
        with torch.no_grad():
            zg0_, zg1_ = torch.chunk( self.merge_head( gg ), 2)
            z = [zg1_, zg0_]
            if self.ens_size>2:
                zg_ = 0.5*(zg0_+zg1_)
                z.append( zg_ )
        return f0_, p, z

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
        for xi in range(len(p)):
            p_ = p[xi]
            z_ = z[xi]
            loss += self.criterion( p_, z_ ) / len(p)
        
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
            [self.backbone, self.prediction_head]
        )
        optimizer = SGD(
            [
                {"name": "simplr", "params": params},
                {"name": "proj", "params": self.projection_head.parameters()},
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
            weight_decay=1e-4,
        )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=0,
                # int(
                #     self.trainer.estimated_stepping_batches
                #     / self.trainer.max_epochs
                #     * 10
                # ),
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
transform = DINOTransform(global_crop_scale=(0.2, 1), local_crop_scale=(0.05, 0.2), n_local_views=n_local_views)
