import copy
from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import SGD, AdamW

from pytorch_lightning import LightningModule

from lightly.loss import NegativeCosineSimilarity
from lightly.models.utils import (
    get_weight_decay_parameters,
    deactivate_requires_grad, 
    update_momentum,
    update_drop_path_rate
)

from lightly.utils.scheduler import CosineWarmupScheduler, cosine_schedule

from linear_classifier import OnlineLinearClassifier

from utils import (
    backbones,
    effective_rank
)

class SimPLR(LightningModule):
    def __init__(self, batch_size_per_device: int,
                 num_classes: int, 
                 warmup: int = 0,
                 backbone:str = "resnet-50",
                 lr:float = 1.0, 
                 decay:float=1e-5,
                 random_head:bool=False,
                 no_projection_head:bool=False,
                 prd_width:int = 256,
                 prj_depth:int = 2,
                 prj_width:int = 2048,
                 no_buttress:bool=False,
                 no_relu:bool=False,
                 no_student_head:bool=False,
                 fwd_multi_crop:bool=False,
                 JS:bool=False, 
                 ema:bool=False,                  
                 AdamW:bool=False,
                 linear_lr:float=0.005,
                 ) -> None:
        super().__init__()
        self.save_hyperparameters('batch_size_per_device',
                                  'num_classes', 
                                  'backbone',
                                  'lr', 'decay', 'warmup',                                  
                                  'random_head',
                                  'no_projection_head',
                                  'no_student_head',
                                  'prd_width', 
                                  "prj_depth", "prj_width",
                                  'no_buttress',
                                  'JS', 'ema',                                   
                                  'AdamW')

        self.warmup = warmup
        self.lr = lr        
        self.linear_lr = linear_lr
        self.decay = decay
        self.batch_size_per_device = batch_size_per_device
        self.JS = JS        
        self.ema = ema
        self.prd_width = prd_width        
        self.AdamW = AdamW
        self.fwd_multi_crop = fwd_multi_crop
        
        self.backbone, emb_width = backbones(backbone)
        self.emb_width = emb_width #needed for the probes

        if no_projection_head:
            self.projection_head = nn.Sequential(nn.Identity())
            prj_width = emb_width
        else:            
            self.projection_head = nn.Sequential(nn.Linear(emb_width, prj_width, bias=False))
            for i in range(prj_depth):
                self.projection_head.extend(
                                        [nn.BatchNorm1d(prj_width),
                                         nn.ReLU(),
                                         nn.Linear(prj_width, prj_width, bias=(i==prj_depth-1))])
        
        if no_student_head:
            self.student_head = nn.Sequential(nn.AdaptiveAvgPool1d(prd_width))
        else:
            self.student_head = nn.Sequential(nn.Linear(prj_width, prd_width, bias=False))
        
        if random_head:
            self.teacher_head = nn.Sequential(nn.Linear(prj_width, prd_width, bias=False))
            nn.init.orthogonal_(self.teacher_head[0].weight)
        elif prj_width > prd_width:
            self.teacher_head = nn.Sequential(nn.AdaptiveAvgPool1d(prd_width))
        elif prj_width == prd_width:
            self.teacher_head = nn.Sequential(nn.Identity())
        else:
            raise NotImplementedError("Invalid Arguments, can't select prd width larger than prj width")        
        
        if not no_relu:
            self.teacher_head.insert(0, nn.ReLU())
            self.student_head.insert(0, nn.ReLU())
        
        if not no_buttress: #Use Batchnorm non-affine for centering
            self.teacher_head.insert(0, nn.BatchNorm1d(prj_width, affine=False))
        
        deactivate_requires_grad(self.teacher_head)
        if self.ema:
            self.teacher_backbone = copy.deepcopy(self.backbone)
            deactivate_requires_grad(self.teacher_backbone)
            self.teacher_projection_head = copy.deepcopy(self.projection_head)
            deactivate_requires_grad(self.teacher_projection_head)
        
        #changed after the copy for ema.
        if backbone in ["vit-s/8", "vit-s/16"]:
            update_drop_path_rate(
                self.backbone.vit,
                drop_path_rate=0.1,  # we recommend using smaller rates like 0.1 for vit-s-14
                mode="uniform",
            )
            if self.ema:
                self.teacher_backbone.eval()


        self.online_classifier = OnlineLinearClassifier(feature_dim=emb_width, num_classes=num_classes)

        self.criterion = NegativeCosineSimilarity()

        # self.automatic_optimization = False

    def forward(self, x: Tensor) -> Tensor:
        if self.ema:
            return self.teacher_backbone(x)
        else:
            return self.backbone(x)
    
    def forward_student(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        h = self.backbone( x ).flatten(start_dim=1)
        z = self.projection_head( h )
        p = self.student_head( z )
        return p, h.detach()
    
    @torch.no_grad()
    def forward_teacher(self, x: Tensor) -> Tensor:
        if self.ema:
            h = self.teacher_backbone( x ).flatten(start_dim=1)
            z = self.teacher_projection_head( h ).detach()    
        else:    
            h = self.backbone( x ).flatten(start_dim=1)
            z = self.projection_head( h )
        q = self.teacher_head( z )
        return q

    def JamesStein(self, diff: Tensor, mean: Tensor, numerator) -> Tensor:
        norm = torch.linalg.vector_norm(diff, dim=1, keepdim=True)
        js_plus = torch.maximum(1.0-numerator/((norm**2)+1e-9), torch.tensor(0.0))
        self.log_dict({"JS_plus":js_plus.mean()})
        q = js_plus*diff + mean
        return q

    def forward_JS(self, x: List[Tensor], idx: Tensor) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        # Two globals
        q0_ = self.forward_teacher(x[0])
        q1_ = self.forward_teacher(x[1])
        q = []

        if self.JS: # For James-Stein
            if len(x) > 2: 
                if self.fwd_multi_crop: 
                    qfwds = [self.forward_teacher(x_) for x_ in x[2:]]
                    mean_ = torch.mean(torch.stack(qfwds, dim=0), dim=0)
                else: # MultiCrops                                    
                    q01_ = 0.5*(q0_+q1_)
                    q.extend([q01_ for _ in range(len(x)-2)])
                    mean_ = self.embedding[idx]
            elif self.fwd_multi_crop:
                raise NotImplementedError("Invalid fwd multicrop no extra views")
            else:
                mean_ = self.embedding[idx]

            # https://openaccess.thecvf.com/content/WACV2024/papers/Khoshsirat_Improving_Normalization_With_the_James-Stein_Estimator_WACV_2024_paper.pdf
            if self.trainer.current_epoch > 0: 
                qdiff0_ = q0_ - mean_
                qdiff1_ = q1_ - mean_
                var_ = torch.mean( (qdiff0_*qdiff1_).abs() )
                numerator_ = (self.prd_width-2.0)*var_
                self.log_dict({"var":var_})
                
                q0_ = self.JamesStein(qdiff0_, mean_, numerator_)
                q1_ = self.JamesStein(qdiff1_, mean_, numerator_)

            self.embedding[idx] = (0.5*(q0_+q1_)).to(torch.float32)
        
        q.insert(0, q0_)
        q.insert(0, q1_)        
        return q

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
        
        # opt = self.optimizers()
        # opt.zero_grad()

        if self.ema: #These lines give us EMA
            momentum = cosine_schedule(self.global_step, self.trainer.estimated_stepping_batches, 0.996, 1)
            update_momentum(self.backbone, self.teacher_backbone, m=momentum)
            update_momentum(self.projection_head, self.teacher_projection_head, m=momentum)
        
        q = self.forward_JS( x, idx )

        loss_sum = 0        
        for xi in range(len(q)):            
            p_, h0_ = self.forward_student(x[xi])
            loss = self.criterion( p_, q[xi] ) / len(q)
            # self.manual_backward(loss)
            loss_sum += loss.detach()

        self.log_dict(
            {"train_loss": loss_sum},
            prog_bar=True,
            sync_dist=True,
            batch_size=len(targets),
        )

        # Online classification.
        cls_loss, cls_log = self.online_classifier.training_step(
            (h0_, targets), batch_idx
        )        
        # self.manual_backward(cls_loss)
        self.log_dict(
            cls_log, 
            sync_dist=True, 
            batch_size=len(targets))
        # opt.step()
        # sch = self.lr_schedulers()
        # sch.step()
        return loss + cls_loss


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
                      on_epoch=True,
                      batch_size=len(targets))

        z = self.projection_head( features )
        q = self.teacher_head( z )

        self.log_dict(
            {"butt_e_rank":effective_rank(z.to(torch.float32))[0]},
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
                            "weight_decay": 1e-7,
                            "lr": self.linear_lr * self.batch_size_per_device * self.trainer.world_size / 256,
                        },
                    ]
        if self.AdamW:      
            optimizer = AdamW(param_cfg,
                lr=self.lr * self.batch_size_per_device * self.trainer.world_size / 256,
                weight_decay=0.0,
            )
        else:  
            optimizer = SGD(param_cfg,
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



