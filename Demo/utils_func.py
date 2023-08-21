import torch
import numpy as np
import os
import random


def single2uint(image):
    '''Numpy:(0,1) (h,w,c) float32 -> Numpy:(0,255) (h,w,c) uint8'''
    return np.uint8((image.clip(0., 1.) * 255.).round())


def uint2single(image):
    '''Numpy:(0,255) (h,w,c) uint8 -> Numpy:(0,1) (h,w,c) float32'''
    return np.float32(image / 255.)


def np2tensor(imgs, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img ,float32):
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, float32) for img in imgs]
    else:
        return _totensor(imgs, float32)


def tensor2np(imgs):
    def _tonp(img):
        img = img.data.cpu().numpy().squeeze(0).transpose(1, 2, 0).astype(np.float32)
        img = np.uint8((img.clip(0., 1.) * 255.).round())
        return img

    if isinstance(imgs, list):
        return [_tonp(img) for img in imgs]
    else:
        return _tonp(imgs)


def denormalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    if type(img) == np.ndarray:  # (h,w,c) numpy (0,255)
        img *= std
        img += mean
    if type(img) == torch.Tensor:  # (c,h,w) tensor
        mean = torch.as_tensor(mean)[None,:, None, None]
        std = torch.as_tensor(std)[None,:, None, None]
        img = img.mul_(std).add_(mean)
    return img


def normalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    if type(img) == np.ndarray:  # (h,w,c) numpy (0,255)
        img /= 255.
        img -= mean
        img /= std
    if type(img) == torch.Tensor:  # (c,h,w) tensor
        mean = torch.as_tensor(mean)[None, :, None, None]
        std = torch.as_tensor(std)[None, :, None, None]
        img = img.sub_(mean).div_(std)
    return img


def _random_crop_patch(img, patch_size):
    if type(img) == np.ndarray:
        h,w,c=img.shape
        h_space = random.randint(0, h - patch_size)
        w_space = random.randint(0, w - patch_size)
        return img[h_space:h_space + patch_size, w_space:w_space + patch_size, :]
    elif type(img) == torch.Tensor:
        b, c, h, w = img.shape
        h_space = random.randint(0, h - patch_size)
        w_space = random.randint(0, w - patch_size)
        return img[:, :, h_space:h_space + patch_size, w_space:w_space + patch_size]
    else:
        w,h=img.size
        h_space = random.randint(0, h - patch_size)
        w_space = random.randint(0, w - patch_size)
        return img.crop((w_space,h_space,w_space+patch_size,h_space+patch_size))




