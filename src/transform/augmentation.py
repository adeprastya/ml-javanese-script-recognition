import numpy as np
import cv2
import albumentations as A


def random_padding(img, **kwargs):
    if img.dtype != np.uint8:
        if img.max() <= 1.1:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    if img.ndim == 2:
        FILL = 255
    else:
        FILL = (255, 255, 255)

    MAX_HORIZONTAL_PAD = 20
    MAX_VERTICAL_PAD = 10

    pl = np.random.randint(0, MAX_HORIZONTAL_PAD + 1)
    pr = np.random.randint(0, MAX_HORIZONTAL_PAD + 1)
    pt = np.random.randint(0, MAX_VERTICAL_PAD + 1)
    pb = np.random.randint(0, MAX_VERTICAL_PAD + 1)

    return cv2.copyMakeBorder(img, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=FILL)


def augmentation_transform(prob=1.0, seed=None):
    return A.Compose(
        [
            A.Lambda(image=random_padding, p=0.5),
            A.OneOf(
                [
                    A.Morphological(scale=[3, 5], operation="dilation", p=1.0),
                    A.Morphological(scale=[3, 5], operation="erosion", p=1.0),
                ],
                p=0.5,
            ),
            A.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                rotate=(-2, 2),
                shear=(-3, 3),
                fit_output=True,
                border_mode=cv2.BORDER_CONSTANT,
                fill=(255, 255, 255),
                p=0.5,
            ),
            A.GaussNoise(std_range=(0.02, 0.1), mean_range=(-0.2, 0.2), p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.5, 2.5), p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.3, 0.3),
                contrast_limit=(-0.3, 0.3),
                ensure_safe_range=True,
                p=0.5,
            ),
        ],
        p=prob,
        seed=seed,
    )


def _dep_all_augmentation_transform(prob=1.0, seed=None):
    return A.Compose(
        [
            A.Lambda(image=random_padding, p=0.6),
            A.OneOf(
                [
                    A.Morphological(scale=[3, 5], operation="dilation", p=1.0),
                    A.Morphological(scale=[3, 5], operation="erosion", p=1.0),
                ],
                p=0.3,
            ),
            A.OneOf(
                [
                    A.ElasticTransform(
                        alpha=10,
                        sigma=60,
                        same_dxdy=True,
                        border_mode=cv2.BORDER_CONSTANT,
                        fill=(255, 255, 255),
                        p=1.0,
                    ),
                    A.GridDistortion(
                        num_steps=5,
                        distort_limit=[-0.1, 0.1],
                        normalized=True,
                        border_mode=cv2.BORDER_CONSTANT,
                        fill=(255, 255, 255),
                        p=1.0,
                    ),
                    A.ThinPlateSpline(
                        scale_range=[0.02, 0.1],
                        num_control_points=4,
                        border_mode=cv2.BORDER_CONSTANT,
                        fill=(255, 255, 255),
                        p=1.0,
                    ),
                ],
                p=0.3,
            ),
            A.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                rotate=(-2, 2),
                shear=(-3, 3),
                fit_output=True,
                border_mode=cv2.BORDER_CONSTANT,
                fill=(255, 255, 255),
                p=0.6,
            ),
            A.GaussNoise(std_range=(0.05, 0.25), mean_range=(-0.3, 0.3), p=0.3),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.5, 2.5), p=1.0),
                    A.Downscale(scale_range=(0.75, 0.95), p=1.0),
                ],
                p=0.6,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.4, 0.4),
                contrast_limit=(-0.4, 0.4),
                ensure_safe_range=True,
                p=0.6,
            ),
        ],
        p=prob,
        seed=seed,
    )
