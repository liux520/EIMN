import torch
import torch.nn as nn
import os
from PIL import Image
import cv2
from tqdm import tqdm
import random
import numpy as np
import torchvision.transforms as tr
from torch.utils.data import Dataset, DataLoader
from natsort import natsorted
from Data.data_utils import _get_paths_from_images


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def uint2tensor(img):
    img = torch.from_numpy(img.astype(np.float32).transpose(2, 0, 1) / 255.).float().unsqueeze(0)
    return img


def tensor2uint8(img):
    img = img.detach().cpu().numpy().astype(np.float32).squeeze(0).transpose(1, 2, 0)
    img = np.uint8((img.clip(0., 1.) * 255.).round())
    return img


if __name__ == '__main__':
    """
    You only need to modify the scale (2/3/4) and input_path (directory or single-image path)
    """
    scale = 2

    save_path = os.path.join(os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)), 'outputs')
    check_dir(save_path)

    # ----------- device ---------- #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------- model ----------- #
    from Model.model_import import model_import

    model = model_import('EIMN_L', scale=scale, load_weights=True).to(device)
    model.eval()

    # input path: directory or single-image path
    input_path = r'E:\Dataset\Restoration\SR\Benchmark\Set5\LRbicx4'

    if os.path.isdir(input_path):
        lr_imgs = _get_paths_from_images(input_path)
    else:
        lr_imgs = [input_path]

    with torch.no_grad():
        for im in tqdm(lr_imgs):
            base, ext = os.path.splitext(os.path.basename(im))
            lr = uint2tensor(cv2.imread(im)[:, :, ::-1]).to(device)
            output = model(lr)
            sr = tensor2uint8(output)
            cv2.imwrite(os.path.join(save_path, f'{base}{ext}'), sr[:, :, ::-1])
