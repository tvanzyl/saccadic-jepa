import copy
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Identity
from torch.optim import SGD
from torch.optim.optimizer import Optimizer
from torchvision.models import resnet50, resnet18

from lightly.loss import NegativeCosineSimilarity
from lightly.models.utils import (
    get_weight_decay_parameters,
)

from lightly.transforms import DINOTransform
from lightly.utils.benchmarking import OnlineLinearClassifier
from lightly.utils.scheduler import CosineWarmupScheduler, cosine_schedule
from SimplRSiam import L2NormalizationLayer

class SimPLR(LightningModule):
    def __init__(self, batch_size_per_device: int, num_classes: int, resnetsize:int = 50) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.batch_size_per_device = batch_size_per_device

        if resnetsize == 18:
            resnet = resnet18()
        else:
            resnet = resnet50()       
        
        emb_width = resnet.fc.in_features
        resnet.fc = Identity()  # Ignore classification head       
        
        self.emb_width = emb_width
        self.upd_width = upd_width = 2048
        self.prd_width = prd_width = emb_width
        self.ens_size = 8

        self.backbone = resnet

        self.projection_head = nn.Sequential(
                nn.Linear(emb_width, upd_width),
                nn.BatchNorm1d(upd_width),
                nn.ReLU(inplace=True),
                nn.Linear(upd_width, prd_width),                
                L2NormalizationLayer(),
                nn.BatchNorm1d(prd_width, affine=False),
                nn.LeakyReLU(),
            )        
        self.prediction_head = nn.Linear(prd_width, prd_width, False)
        self.merge_head = nn.Linear(prd_width, prd_width)
        self.merge_head.weight.data = self.prediction_head.weight.data
        self.criterion = NegativeCosineSimilarity()

        self.online_classifier = OnlineLinearClassifier(feature_dim=emb_width, num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def forward_student(self, x: Tensor) -> Tensor:
        p, g, z = [], [], []
        #Pass Through Each Global Seperate backwards cause we want the first one for online linear
        for i in range(1,-1,-1):
            f_ = self.backbone( x[i] ).flatten(start_dim=1)
            g_ = self.projection_head( f_ )
            g.append( g_.detach() )
            p_ = self.prediction_head( g_ )
            p.append( p_ )

        #Pass Through The Locals Together
        if self.ens_size > 2:
            x__ = torch.cat( x[2:] )
            f__ = self.backbone( x__ ).flatten(start_dim=1)
            g__ = self.projection_head( f__ )
            p__ = self.prediction_head( g__ )
            p.extend( p__.chunk(self.ens_size-2) )
        
        # Create The Teacher Weighted Equal To Globals and Locals
        with torch.no_grad():            
            e_ = torch.stack(g, dim=1).mean(dim=1)
            zg_ = self.merge_head( e_ )
            e__ = g__.detach().view(-1,self.batch_size_per_device,self.prd_width).mean(dim=0)
            zl_ = self.merge_head( e__ )
            z_ = torch.stack([zg_, zl_], dim=1).mean(dim=1)
        z.extend([z_,z_])
        for i in range(self.ens_size-2):
            z.append( zg_ )

        return f_.detach(), p, z

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
            [self.backbone, self.projection_head, self.prediction_head]
        )
        optimizer = SGD(
            [
                {"name": "simplr", "params": params},
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
            lr=0.03 * self.batch_size_per_device * self.trainer.world_size / 256,
            momentum=0.9,
            weight_decay=1e-4,
        )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=0,
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
        # self.student_projection_head.cancel_last_layer_gradients(self.current_epoch)

transform = DINOTransform(global_crop_scale=(0.2,  1.0),local_crop_scale =(0.08, 0.2))
