import random

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF


class RandomShear:
    def __init__(self, max_angle):
        self.max_angle = max_angle

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        angle = random.uniform(-self.max_angle, self.max_angle)
        return TF.affine(img, angle=0, translate=(0, 0), scale=1, shear=angle)


class RandomBrightness:
    def __init__(self, min_ratio: float = 0.7, max_ratio: float = 1.3) -> None:
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        return img * ratio


class RandomErase:
    def __init__(self, p: float, scale=(0.02, 0.2), ratio=(0.3, 3.3)) -> None:
        self.p = p
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img

        _, h, w = img.shape
        area = h * w
        target_area = random.uniform(self.scale[0], self.scale[1]) * area
        aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])

        h_erase = int(np.sqrt(target_area / aspect_ratio))
        w_erase = int(target_area / h_erase)
        h_erase = max(1, min(h_erase, h))
        w_erase = max(1, min(w_erase, w))

        top = random.randint(0, h - h_erase)
        left = random.randint(0, w - w_erase)

        img[:, top:top + h_erase, left:left + w_erase] = 0
        return img


class GaussianNoise:
    def __init__(self, mean: float = 0.0, std: float = 0.1) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(img) * self.std + self.mean
        return img + noise