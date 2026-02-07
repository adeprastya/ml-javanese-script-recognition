import numpy as np
from PIL import Image, ImageOps
import cv2
import torchvision.transforms as T


class CLAHE:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self._clahe = None

    def __call__(self, img: Image.Image) -> Image.Image:
        if self._clahe is None:
            self._clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size
            )

        img_np = np.array(img)

        # If Grayscale
        if img_np.ndim == 2:
            return Image.fromarray(self._clahe.apply(img_np))

        # If RGB
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        lab = cv2.merge((l, a, b))
        return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


class ResizeByHeight:
    def __init__(self, height):
        self.height = height

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        new_w = int(w * self.height / h)
        return img.resize((new_w, self.height), Image.BILINEAR)


class Invert(object):
    def __call__(self, img):
        return ImageOps.invert(img)


def preprocessing_transform(img_height, enhance=False):
    transforms = [
        T.Grayscale(1),
    ]

    if enhance:
        transforms.append(CLAHE())

    transforms.extend(
        [
            Invert(),
            ResizeByHeight(img_height),
            T.ToTensor(),
            T.Normalize(mean=[0.0], std=[1.0]),  # [0, 1]
        ]
    )

    return T.Compose(transforms)
