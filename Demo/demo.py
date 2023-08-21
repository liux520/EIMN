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
    scale = 2

    save = True
    save_path = os.path.join(os.path.abspath(os.path.join(__file__, os.path.pardir)), 'outputs')
    inputs_path = os.path.join(os.path.abspath(os.path.join(__file__, os.path.pardir)), 'inputs')
    check_dir(save_path)

    # ----------- device ---------- #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------- model ----------- #
    from Model.model_import import model_import

    model = model_import('EIMN_L', scale=scale, load_weights=True).to(device)
    model.eval()

    lr_imgs = _get_paths_from_images(inputs_path)
    # lr_imgs = ''
    # if not isinstance(lr_imgs, (list, tuple)):
    #     lr_imgs = [lr_imgs]
    with torch.no_grad():
        for im in tqdm(lr_imgs):
            base, ext = os.path.splitext(os.path.basename(im))
            lr = uint2tensor(cv2.imread(im)[:, :, ::-1]).to(device)
            output = model(lr)
            sr = tensor2uint8(output)
            cv2.imwrite(os.path.join(save_path, f'{base}{ext}'), sr[:, :, ::-1])
