import torch
from torchvision import transforms
from CMRSegment.config import AugmentationConfig
import numpy as np
from typing import Tuple
from scipy.ndimage import zoom
from scipy.spatial.transform import Rotation


def resize_image(image: np.ndarray, target_shape: Tuple, order: int):
    image_shape = image.shape
    factors = [float(target_shape[i]) / image_shape[i] for i in range(len(image_shape))]
    output = zoom(image, factors, order=order)
    return output


def random_crop(image: np.ndarray, label: np.ndarray, output_size: Tuple):
    """
    image size = (slice, weight, height)
    """
    slice, weight, height = image.shape
    s, w, h = output_size

    i = np.random.randint(0, slice - s)
    j = np.random.randint(0, weight - w)
    k = np.random.randint(0, height - h)

    cropped_image = image[i: i + s, j: j + w, k: k + h]
    cropped_label = label[:, i: i + s, j: j + w, k: k + h]

    return cropped_image, cropped_label


def random_flip(image: np.ndarray, label: np.ndarray, flip_prob: float):
    for axis in range(0, 3):
        if np.random.rand() >= flip_prob:
            image = np.flip(image, axis=axis)
            label = np.flip(label, axis=axis + 1)
    return image, label


def random_rotation(image: np.ndarray, label: np.ndarray, angles: Tuple[float]):
    rotation_angles = []
    for idx, angle in enumerate(angles):
        rotation_angles.append(np.random.uniform(-angle, angle))
    rotation = Rotation.from_euler("xyz", angles=rotation_angles, degrees=True)
    image = rotation.apply(image)
    # How is it interpolated? Rounding?
    label = rotation.apply(label)
    return image, label


def random_scaling(image: np.ndarray, label: np.ndarray, delta_factors: Tuple[float]):
    """delta_factor = (0.2, 0.2, 0.2), which leads to scale factors of (1+-0.2, 1+-0.2, 1+-0.2)"""
    factors = []
    for idx, delta in enumerate(delta_factors):
        factors.append(np.random.uniform(1 - delta, 1 + delta))
    image = zoom(image, factors, order=1)
    labels = []
    for i in range(label.shape[0]):
        labels.append(zoom(label[i, :, :, :], factors, order=0))
    label = np.stack(labels, axis=0)
    return image, label


def random_brightness(image, max_delta):
    delta = np.random.uniform(-max_delta, max_delta)
    image = image + delta
    return image


def random_contrast(image, delta):
    lower = 1 - delta
    upper = 1 + delta
    contrast_factor = np.random.uniform(lower, upper)
    mean = np.mean(image)
    image = (image - mean) * contrast_factor + mean
    return image


def adjust_gamma(image, delta):
    gamma = np.random.uniform(1 - delta, 1 + delta)
    # image = 1 * image ** gamma
    image = np.power(image, gamma)
    return image


def random_channel_shift(image, brightness, contrast, gamma):
    image = random_brightness(image, brightness)
    image = random_contrast(image, contrast)
    image = adjust_gamma(image, gamma)
    image = np.clip(image, 0, 1)
    return image


def augment(image: np.ndarray, label: np.ndarray, config: AugmentationConfig, output_size, seed: int = None):
    """image = (slice, weight, height), label = (class, slice, weight, height)"""
    if seed is None:
        seed = np.random.randint(0, 10000000)
    np.random.seed(seed)
    image = zoom(image, (1 + config.scaling_factors[0], 1 + config.scaling_factors[1], 1 + config.scaling_factors[2]), order=1)
    labels = []
    for i in range(label.shape[0]):
        labels.append(zoom(label[i, :, :, :], (1 + config.scaling_factors[0], 1 + config.scaling_factors[1], 1 + config.scaling_factors[2]), order=0))
    label = np.stack(labels, axis=0)
    image, label = random_flip(image, label, config.flip)
    # image, label = random_rotation(image, label, config.rotation_angles)
    image, label = random_scaling(image, label, config.scaling_factors)
    image, label = random_crop(image, label, output_size)
    if config.channel_shift:
        image = random_channel_shift(image, config.brightness, config.contrast, config.gamma)
    return image, label
