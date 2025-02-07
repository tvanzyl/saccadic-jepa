import torch
import torch.nn as nn
from torch.nn import functional as F

from lightly.transforms import (    
    DINOTransform
    
)

# Use Multi-Crop augmentations https://arxiv.org/html/2403.05726v1#bib.bib7
SIMSIMPTRansform = {
32:DINOTransform(global_crop_size=32,
                 global_crop_scale=(0.2, 1.0),
                 n_local_views=0,
                 gaussian_blur=(0.0, 0.0, 0.0),
                ),
64:DINOTransform(global_crop_size=64,
                 global_crop_scale=(0.2, 1.0),
                 local_crop_size=32,
                 local_crop_scale=(0.08, 0.2),
                 gaussian_blur=(0.0, 0.0, 0.0),
                ),
96:DINOTransform(global_crop_size=96,
                 global_crop_scale=(0.4, 1.0),
                 local_crop_size=48,
                 local_crop_scale=(0.08, 0.4),
                ),
128:DINOTransform(global_crop_size=128,
                  global_crop_scale=(0.2, 1.0),
                  local_crop_size=64,
                  local_crop_scale=(0.08, 0.2),
                ),
244:DINOTransform(global_crop_size=224,
                  global_crop_scale=(0.2,  1.0),
                  local_crop_scale =(0.08, 0.2),
                ),
}

class L2NormalizationLayer(nn.Module):
    def __init__(self, dim=1, eps=1e-12):
        super(L2NormalizationLayer, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim, eps=self.eps)


# import torchvision
# torchvision.datasets.STL10(root="/data/stl10", download=True, split='train+unlabeled')
# torchvision.datasets.STL10(root="/data/stl10", split='test')
