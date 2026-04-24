import torch
import torch.nn as nn

from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl

from torchvision.transforms import v2 as T
from torchvision.models import resnet50, resnet34 , resnet18

from timm.models.vision_transformer import vit_small_patch16_224, vit_small_patch8_224

from lightly.models.modules import MaskedVisionTransformerTIMM
from lightly.transforms import DINOTransform
from lightly.transforms.utils import IMAGENET_NORMALIZE

class OverrideEpochStepCallback(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def _log_step_as_current_epoch(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        pl_module.log("step", trainer.current_epoch)


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

class MoCoVisionTransformerTIMM(MaskedVisionTransformerTIMM):
    def __init__(self, vit, mask_token = None, weight_initialization = "", antialias = True, pos_embed_initialization = "sincos", CLScount=4):
        super().__init__(vit, mask_token, weight_initialization, antialias, pos_embed_initialization)
        for param in vit.patch_embed.parameters():
            param.requires_grad = False
        self.CLScount = CLScount
        self.emb_width = vit.embed_dim*CLScount

    def forward(self, images):
        out, intermediates = self.forward_intermediates(images, norm=True)
        return torch.stack(intermediates[-self.CLScount:])[:, :, 0].transpose(0,1).flatten(start_dim=1)

def backbones(name):
    if name in ["resnetjie-18"]:
        resnet = {"resnetjie-18":resnet18}[name]()
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
    elif name in ["vit-s/8", "vit-s/16"]:
        vit = {"vit-s/16":vit_small_patch16_224,
               "vit-s/8": vit_small_patch8_224}[name](dynamic_img_size=True)
        mvt = MoCoVisionTransformerTIMM(vit=vit)
        emb_width = mvt.emb_width
        backbone = mvt
    else:
        raise NotImplemented("Backbone Not Supported")
    print(f"Emb Width {emb_width}")
    return backbone, emb_width


CIFAR10_NORMALIZE   = {'mean':(0.4914, 0.4822, 0.4465), 'std':(0.2470, 0.2435, 0.2616)}
CIFAR100_NORMALIZE  = {'mean':(0.5071, 0.4867, 0.4408), 'std':(0.2675, 0.2565, 0.2761)}
TINYIMAGE_NORMALIZE = {'mean':(0.4802, 0.4481, 0.3975), 'std':(0.2302, 0.2265, 0.2262)}
STL10_NORMALIZE     = {'mean':(0.4408, 0.4279, 0.3867), 'std':(0.2682, 0.2610, 0.2686)}

train_identity= lambda NORMALIZE: T.Compose([                    
                    T.RandomHorizontalFlip(), T.ToImage(),  T.ToDtype(torch.float32, scale=True),
                    T.Normalize(mean=NORMALIZE["mean"], std=NORMALIZE["std"]),
                ])
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
"Cifar100-4":   DINOTransform(global_crop_size=32,
                            global_crop_scale=(0.08, 1.0),
                            n_local_views=2,
                            local_crop_size=32,
                            local_crop_scale=(0.08, 1.0),
                            gaussian_blur=(0.5, 0.0, 0.5),
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

METHODS = {
    "Cifar10-2":    {"train_transform": train_transforms["Cifar10"],  
                     "val_transform": val_transforms["Cifar10"], 
                     "transform": transforms["Cifar10-2"],},

    "Cifar100-2":   {"train_transform": train_transforms["Cifar100"],  
                     "val_transform": val_transforms["Cifar100"], 
                     "transform": transforms["Cifar100-2"],},
    "Cifar100-4":   {"train_transform": train_transforms["Cifar100"],  
                     "val_transform": val_transforms["Cifar100"], 
                     "transform": transforms["Cifar100-4"],},


    "Tiny-2":       {"train_transform": train_transforms["Tiny"],  
                     "val_transform": val_transforms["Tiny"],
                     "transform": transforms["Tiny-2"],},

    "STL-2":        {"train_transform": train_transforms["STL"],  
                     "val_transform": val_transforms["STL"],
                     "transform": transforms["STL-2"],},

    "Im100-2":      {"train_transform": train_transforms["Im100"],
                     "val_transform": val_transforms["Im100"],
                     "transform": transforms["Im100-2"],},
    "Im100-8":      {"train_transform": train_transforms["Im100"],
                     "val_transform": val_transforms["Im100"],
                     "transform": transforms["Im100-8"],},

    "Im1k-2":       {"train_transform": train_transforms["Im1k"],
                     "val_transform": val_transforms["Im1k"],
                     "transform": transforms["Im1k-2"],},
    "Im1k-8":       {"train_transform": train_transforms["Im1k"],
                     "val_transform": val_transforms["Im1k"],
                     "transform": transforms["Im1k-8"],},
}