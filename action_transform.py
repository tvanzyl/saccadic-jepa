from typing import Dict, List, Optional, Tuple, Union

import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL.Image import Image
from torch import Tensor, rand, sigmoid

from lightly.transforms.simsiam_transform import SimSiamViewTransform
from lightly.transforms.gaussian_blur import GaussianBlur
from lightly.transforms.multi_view_transform import MultiViewTransform
from lightly.transforms.rotation import random_rotation_transform
from lightly.transforms.utils import IMAGENET_NORMALIZE

from PIL import ImageFilter
import numpy as np

class ActionTransform(MultiViewTransform):
    """Implements the transformations for SimSiam.

    Input to this transform:
        PIL Image or Tensor.

    Output of this transform:
        List of Tensor of length 2.

    Applies the following augmentations by default:
        - Random resized crop
        - Random horizontal flip
        - Color jitter
        - Random gray scale
        - Gaussian blur
        - ImageNet normalization

    Attributes:
        input_size:
            Size of the input image in pixels.
        cj_prob:
            Probability that color jitter is applied.
        cj_strength:
            Strength of the color jitter. `cj_bright`, `cj_contrast`, `cj_sat`, and
            `cj_hue` are multiplied by this value. For datasets with small images,
            such as CIFAR, it is recommended to set `cj_strength` to 0.5.
        cj_bright:
            How much to jitter brightness.
        cj_contrast:
            How much to jitter constrast.
        cj_sat:
            How much to jitter saturation.
        cj_hue:
            How much to jitter hue.
        min_scale:
            Minimum size of the randomized crop relative to the input_size.
        random_gray_scale:
            Probability of conversion to grayscale.
        gaussian_blur:
            Probability of Gaussian blur.
        kernel_size:
            Will be deprecated in favor of `sigmas` argument. If set, the old behavior applies and `sigmas` is ignored.
            Used to calculate sigma of gaussian blur with kernel_size * input_size.
        sigmas:
            Tuple of min and max value from which the std of the gaussian kernel is sampled.
            Is ignored if `kernel_size` is set.
        vf_prob:
            Probability that vertical flip is applied.
        hf_prob:
            Probability that horizontal flip is applied.
        rr_prob:
            Probability that random rotation is applied.
        rr_degrees:
            Range of degrees to select from for random rotation. If rr_degrees is None,
            images are rotated by 90 degrees. If rr_degrees is a (min, max) tuple,
            images are rotated by a random angle in [min, max]. If rr_degrees is a
            single number, images are rotated by a random angle in
            [-rr_degrees, +rr_degrees]. All rotations are counter-clockwise.
        normalize:
            Dictionary with 'mean' and 'std' for torchvision.transforms.Normalize.

    """

    def __init__(
        self,
        input_size: int = 224,
        cj_prob: float = 0.8,
        cj_strength: float = 1.0,
        cj_bright: float = 0.4,
        cj_contrast: float = 0.4,
        cj_sat: float = 0.4,
        cj_hue: float = 0.1,
        min_scale: float = 0.2,
        random_gray_scale: float = 0.2,
        gaussian_blur: float = 0.5,
        kernel_size: Optional[float] = None,
        sigmas: Tuple[float, float] = (0.1, 2),
        vf_prob: float = 0.0,
        hf_prob: float = 0.5,
        rr_prob: float = 0.0,
        rr_degrees: Optional[Union[float, Tuple[float, float]]] = None,
        normalize: Union[None, Dict[str, List[float]]] = IMAGENET_NORMALIZE,
    ):
        view_transform = ActionViewTransform(
            input_size=input_size,
            cj_prob=cj_prob,
            cj_strength=cj_strength,
            cj_bright=cj_bright,
            cj_contrast=cj_contrast,
            cj_sat=cj_sat,
            cj_hue=cj_hue,
            min_scale=min_scale,
            random_gray_scale=random_gray_scale,
            gaussian_blur=gaussian_blur,
            kernel_size=kernel_size,
            sigmas=sigmas,
            vf_prob=vf_prob,
            hf_prob=hf_prob,
            rr_prob=rr_prob,
            rr_degrees=rr_degrees,
            normalize=normalize,
        )

        identity_transform = T.Compose([T.ToTensor(),
                                        T.Normalize(mean=IMAGENET_NORMALIZE["mean"],
                                                    std=IMAGENET_NORMALIZE["std"])])

        super().__init__(transforms=[identity_transform, view_transform, view_transform, view_transform, view_transform])



class ActionViewTransform:
    def __init__(
        self,
        input_size: int = 224,
        cj_prob: float = 0.8,
        cj_strength: float = 1.0,
        cj_bright: float = 0.4,
        cj_contrast: float = 0.4,
        cj_sat: float = 0.4,
        cj_hue: float = 0.1,
        min_scale: float = 0.2,
        random_gray_scale: float = 0.2,
        gaussian_blur: float = 0.5,
        kernel_size: Optional[float] = None,
        sigmas: Tuple[float, float] = (0.1, 2),
        vf_prob: float = 0.0,
        hf_prob: float = 0.5,
        rr_prob: float = 0.0,
        rr_degrees: Optional[Union[float, Tuple[float, float]]] = None,
        normalize: Union[None, Dict[str, List[float]]] = IMAGENET_NORMALIZE,
    ):
        self.input_size = input_size
        self.cj_prob = cj_prob
        self.cj_strength = cj_strength
        self.cj_bright = cj_bright
        self.cj_contrast = cj_contrast
        self.cj_sat = cj_sat
        self.cj_hue = cj_hue
        self.min_scale = min_scale
        self.random_gray_scale = random_gray_scale
        self.gaussian_blur = gaussian_blur
        self.kernel_size = kernel_size
        self.sigmas = sigmas
        self.vf_prob = vf_prob
        self.hf_prob = hf_prob
        self.rr_prob = rr_prob
        self.rr_degrees = rr_degrees
        self.normalize = normalize

        self.color_jitter = T.ColorJitter(
            brightness=cj_strength * cj_bright,
            contrast=cj_strength * cj_contrast,
            saturation=cj_strength * cj_sat,
            hue=cj_strength * cj_hue,
        )

        self.randomized_crop = T.RandomResizedCrop(size=input_size, scale=(min_scale, 1.0))        
        
        # transform = [
        #     T.RandomResizedCrop(size=input_size, scale=(min_scale, 1.0)),
        #     # random_rotation_transform(rr_prob=rr_prob, rr_degrees=rr_degrees),            
        #     T.RandomHorizontalFlip(p=hf_prob),
        #     T.RandomVerticalFlip(p=vf_prob),
        #     T.RandomApply([color_jitter], p=cj_prob),
        #     T.RandomGrayscale(p=random_gray_scale),
        #     GaussianBlur(kernel_size=kernel_size, sigmas=sigmas, prob=gaussian_blur),
        #     T.ToTensor(),
        # ]
        # if normalize:
        #     transform += [T.Normalize(mean=normalize["mean"], std=normalize["std"])]
        # self.transform = T.Compose(transform)

    def __call__(self, image: Union[Tensor, Image]) -> Union[Tensor, Tensor]:
        """
        Applies the transforms to the input image.

        Args:
            image:
                The input image to apply the transforms to.

        Returns:
            The transformed image.

        """                
        rands = rand(7)
        rr = rands[0] < self.rr_prob        
        hf = rands[1] < self.hf_prob
        vf = rands[2] < self.vf_prob
        cj = rands[3] < self.cj_prob
        gs = rands[4] < self.random_gray_scale
        gb = rands[5] < self.gaussian_blur

        top, left, height, width = T.RandomResizedCrop.get_params(image,
                                                                  self.randomized_crop.scale,
                                                                  self.randomized_crop.ratio)                   
        image = TF.resized_crop(image, top, left, height, width,
                                self.randomized_crop.size, 
                                self.randomized_crop.interpolation,
                                self.randomized_crop.antialias)
        if rr:
            if self.rr_degrees is None:
                degrees = 90.0
            else:
                degrees = T.RandomRotation.get_params(self.rr_degrees)
            image = TF.rotate(image, degrees)
        if hf:
            image = TF.hflip(image)
        if vf:
            image = TF.vflip(image)
        if cj:
            fn_idx, brighten, contrast, saturate, hue = T.ColorJitter.get_params(self.color_jitter.brightness,
                                                                                 self.color_jitter.contrast,
                                                                                 self.color_jitter.saturation,
                                                                                 self.color_jitter.hue)
            for fn_id in fn_idx:
                if fn_id == 0 and brighten is not None:
                    image = TF.adjust_brightness(image, brighten)
                elif fn_id == 1 and contrast is not None:
                    image = TF.adjust_contrast(image, contrast)
                elif fn_id == 2 and saturate is not None:
                    image = TF.adjust_saturation(image, saturate)
                elif fn_id == 3 and hue is not None:
                    image = TF.adjust_hue(image, hue)
        if gs:
            image = TF.to_grayscale(image, 3)
        if gb:
            sigma = self.sigmas[0] + rands[6]*(self.sigmas[1]-self.sigmas[0])
            image = image.filter(ImageFilter.GaussianBlur(radius=sigma))
        
        image = TF.to_tensor(image)
        if self.normalize:
            image = TF.normalize(image, self.normalize["mean"], self.normalize["std"]) 

        transformed: Tensor = image
        return transformed, Tensor([top/self.input_size, 
                                    left/self.input_size, 
                                    height/self.input_size, 
                                    width/self.input_size,
                                    rr*degrees/360.0,
                                    hf*1.0, vf*1.0,
                                    cj*fn_idx[0]/4.0, cj*fn_idx[1]/4.0, cj*fn_idx[2]/4.0, cj*fn_idx[3]/4.0,
                                    cj*brighten/self.cj_bright, 
                                    cj*contrast/self.cj_contrast, 
                                    cj*saturate/self.cj_sat, 
                                    cj*hue/self.cj_hue,
                                    gs*1.0,
                                    (gb*sigma-self.sigmas[0])/self.sigmas[1]])


class SimSimPTransform(MultiViewTransform):
    """Implements the transformations for SimSiam.

    Input to this transform:
        PIL Image or Tensor.

    Output of this transform:
        List of Tensor of length 2.

    Applies the following augmentations by default:
        - Random resized crop
        - Random horizontal flip
        - Color jitter
        - Random gray scale
        - Gaussian blur
        - ImageNet normalization

    Attributes:
        input_size:
            Size of the input image in pixels.
        cj_prob:
            Probability that color jitter is applied.
        cj_strength:
            Strength of the color jitter. `cj_bright`, `cj_contrast`, `cj_sat`, and
            `cj_hue` are multiplied by this value. For datasets with small images,
            such as CIFAR, it is recommended to set `cj_strength` to 0.5.
        cj_bright:
            How much to jitter brightness.
        cj_contrast:
            How much to jitter constrast.
        cj_sat:
            How much to jitter saturation.
        cj_hue:
            How much to jitter hue.
        min_scale:
            Minimum size of the randomized crop relative to the input_size.
        random_gray_scale:
            Probability of conversion to grayscale.
        gaussian_blur:
            Probability of Gaussian blur.
        kernel_size:
            Will be deprecated in favor of `sigmas` argument. If set, the old behavior applies and `sigmas` is ignored.
            Used to calculate sigma of gaussian blur with kernel_size * input_size.
        sigmas:
            Tuple of min and max value from which the std of the gaussian kernel is sampled.
            Is ignored if `kernel_size` is set.
        vf_prob:
            Probability that vertical flip is applied.
        hf_prob:
            Probability that horizontal flip is applied.
        rr_prob:
            Probability that random rotation is applied.
        rr_degrees:
            Range of degrees to select from for random rotation. If rr_degrees is None,
            images are rotated by 90 degrees. If rr_degrees is a (min, max) tuple,
            images are rotated by a random angle in [min, max]. If rr_degrees is a
            single number, images are rotated by a random angle in
            [-rr_degrees, +rr_degrees]. All rotations are counter-clockwise.
        normalize:
            Dictionary with 'mean' and 'std' for torchvision.transforms.Normalize.

    """

    def __init__(
        self,
        number_augments:int = 2,
        include_identity:bool = False,
        input_size: int = 224,
        cj_prob: float = 0.8,
        cj_strength: float = 1.0,
        cj_bright: float = 0.4,
        cj_contrast: float = 0.4,
        cj_sat: float = 0.4,
        cj_hue: float = 0.1,
        min_scale: float = 0.2,
        random_gray_scale: float = 0.2,
        gaussian_blur: float = 0.5,
        kernel_size: Optional[float] = None,
        sigmas: Tuple[float, float] = (0.1, 2),
        vf_prob: float = 0.0,
        hf_prob: float = 0.5,
        rr_prob: float = 0.0,
        rr_degrees: Optional[Union[float, Tuple[float, float]]] = None,
        normalize: Union[None, Dict[str, List[float]]] = IMAGENET_NORMALIZE,
    ):
        view_transform = SimSiamViewTransform(
            input_size=input_size,
            cj_prob=cj_prob,
            cj_strength=cj_strength,
            cj_bright=cj_bright,
            cj_contrast=cj_contrast,
            cj_sat=cj_sat,
            cj_hue=cj_hue,
            min_scale=min_scale,
            random_gray_scale=random_gray_scale,
            gaussian_blur=gaussian_blur,
            kernel_size=kernel_size,
            sigmas=sigmas,
            vf_prob=vf_prob,
            hf_prob=hf_prob,
            rr_prob=rr_prob,
            rr_degrees=rr_degrees,
            normalize=normalize,
        )
        identity_transform = T.Compose([T.Resize((input_size,input_size)),
                                        T.ToTensor(),
                                        T.Normalize(mean=IMAGENET_NORMALIZE["mean"],
                                                    std=IMAGENET_NORMALIZE["std"])])
        if include_identity:
            transforms=[identity_transform,]+[view_transform,]*number_augments
        else:
            transforms=[view_transform,]*number_augments
        super().__init__(transforms=transforms)

class RankingTransform(MultiViewTransform):
    """Implements the transformations for SimSiam.

    Input to this transform:
        PIL Image or Tensor.

    Output of this transform:
        List of Tensor of length 2.

    Applies the following augmentations by default:
        - Random resized crop
        - Random horizontal flip
        - Color jitter
        - Random gray scale
        - Gaussian blur
        - ImageNet normalization

    Attributes:
        input_size:
            Size of the input image in pixels.
        cj_prob:
            Probability that color jitter is applied.
        cj_strength:
            Strength of the color jitter. `cj_bright`, `cj_contrast`, `cj_sat`, and
            `cj_hue` are multiplied by this value. For datasets with small images,
            such as CIFAR, it is recommended to set `cj_strength` to 0.5.
        cj_bright:
            How much to jitter brightness.
        cj_contrast:
            How much to jitter constrast.
        cj_sat:
            How much to jitter saturation.
        cj_hue:
            How much to jitter hue.
        min_scale:
            Minimum size of the randomized crop relative to the input_size.
        random_gray_scale:
            Probability of conversion to grayscale.
        gaussian_blur:
            Probability of Gaussian blur.
        kernel_size:
            Will be deprecated in favor of `sigmas` argument. If set, the old behavior applies and `sigmas` is ignored.
            Used to calculate sigma of gaussian blur with kernel_size * input_size.
        sigmas:
            Tuple of min and max value from which the std of the gaussian kernel is sampled.
            Is ignored if `kernel_size` is set.
        vf_prob:
            Probability that vertical flip is applied.
        hf_prob:
            Probability that horizontal flip is applied.
        rr_prob:
            Probability that random rotation is applied.
        rr_degrees:
            Range of degrees to select from for random rotation. If rr_degrees is None,
            images are rotated by 90 degrees. If rr_degrees is a (min, max) tuple,
            images are rotated by a random angle in [min, max]. If rr_degrees is a
            single number, images are rotated by a random angle in
            [-rr_degrees, +rr_degrees]. All rotations are counter-clockwise.
        normalize:
            Dictionary with 'mean' and 'std' for torchvision.transforms.Normalize.

    """

    def __init__(
        self,
        input_size: int = 224,
        cj_prob: float = 0.8,
        cj_strength: float = 1.0,
        cj_bright: float = 0.4,
        cj_contrast: float = 0.4,
        cj_sat: float = 0.4,
        cj_hue: float = 0.1,
        min_scale: float = 0.2,
        random_gray_scale: float = 0.2,
        gaussian_blur: float = 0.5,
        kernel_size: Optional[float] = None,
        sigmas: Tuple[float, float] = (0.1, 2),
        vf_prob: float = 0.0,
        hf_prob: float = 0.5,
        rr_prob: float = 0.0,
        rr_degrees: Optional[Union[float, Tuple[float, float]]] = None,
        normalize: Union[None, Dict[str, List[float]]] = IMAGENET_NORMALIZE,
    ):
        view_transform1 = SimSiamViewTransform(
            input_size=input_size,
            cj_prob=0,
            cj_strength=cj_strength,
            cj_bright=cj_bright,
            cj_contrast=cj_contrast,
            cj_sat=cj_sat,
            cj_hue=cj_hue,
            min_scale=min_scale,
            random_gray_scale=0,
            gaussian_blur=0,
            kernel_size=kernel_size,
            sigmas=sigmas,
            vf_prob=0,
            hf_prob=0,
            rr_prob=0,
            rr_degrees=rr_degrees,
            normalize=normalize,
        )
        view_transform2 = SimSiamViewTransform(
            input_size=input_size,
            cj_prob=0,
            cj_strength=cj_strength,
            cj_bright=cj_bright,
            cj_contrast=cj_contrast,
            cj_sat=cj_sat,
            cj_hue=cj_hue,
            min_scale=min_scale,
            random_gray_scale=random_gray_scale,
            gaussian_blur=gaussian_blur,
            kernel_size=kernel_size,
            sigmas=sigmas,
            vf_prob=0,
            hf_prob=0,
            rr_prob=0,
            rr_degrees=rr_degrees,
            normalize=normalize,
        )
        view_transform3 = SimSiamViewTransform(
            input_size=input_size,
            cj_prob=cj_prob,
            cj_strength=cj_strength,
            cj_bright=cj_bright,
            cj_contrast=cj_contrast,
            cj_sat=cj_sat,
            cj_hue=cj_hue,
            min_scale=min_scale,
            random_gray_scale=random_gray_scale,
            gaussian_blur=gaussian_blur,
            kernel_size=kernel_size,
            sigmas=sigmas,
            vf_prob=0,
            hf_prob=0,
            rr_prob=0,
            rr_degrees=rr_degrees,
            normalize=normalize,
        )
        view_transform4 = SimSiamViewTransform(
            input_size=input_size,
            cj_prob=cj_prob,
            cj_strength=cj_strength,
            cj_bright=cj_bright,
            cj_contrast=cj_contrast,
            cj_sat=cj_sat,
            cj_hue=cj_hue,
            min_scale=min_scale,
            random_gray_scale=random_gray_scale,
            gaussian_blur=gaussian_blur,
            kernel_size=kernel_size,
            sigmas=sigmas,
            vf_prob=vf_prob,
            hf_prob=hf_prob,
            rr_prob=rr_prob,
            rr_degrees=rr_degrees,
            normalize=normalize,
        )
        identity_transform = T.Compose([T.ToTensor(),
                                        T.Normalize(mean=IMAGENET_NORMALIZE["mean"],
                                                    std=IMAGENET_NORMALIZE["std"])])

        super().__init__(transforms=[identity_transform, view_transform1, view_transform2, view_transform3, view_transform4])
