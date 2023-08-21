import os
import random
import cv2
import numpy as np
from PIL import Image
from natsort import natsorted
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import functional as F
from Utils.imresize import imresize
from Data.data_utils import _get_paths_from_images, paired_random_crop_, random_augmentation


def Dataloader(scale, gt_size, train_batchsize, val_batchsize, num_worker, pin_memory=False):
    train_dataset = Dataset_PairedImage(scale, gt_size=gt_size)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batchsize, shuffle=True,
                                  num_workers=num_worker, drop_last=True, pin_memory=pin_memory)

    set5 = Benchmark(dataset='Set5', scale=scale)
    set14 = Benchmark(dataset='Set14', scale=scale)
    urban100 = Benchmark(dataset='Urban100', scale=scale)
    manga109 = Benchmark(dataset='Manga109', scale=scale)

    set5_dataloader = DataLoader(set5, batch_size=val_batchsize, shuffle=False,
                                 num_workers=num_worker, drop_last=False, pin_memory=pin_memory)
    set14_dataloader = DataLoader(set14, batch_size=val_batchsize, shuffle=False,
                                  num_workers=num_worker, drop_last=False, pin_memory=pin_memory)
    urban100_dataloader = DataLoader(urban100, batch_size=val_batchsize, shuffle=False,
                                     num_workers=num_worker, drop_last=False, pin_memory=pin_memory)
    manga109_dataloader = DataLoader(manga109, batch_size=val_batchsize, shuffle=False,
                                     num_workers=num_worker, drop_last=False, pin_memory=pin_memory)

    return train_dataloader, set5_dataloader, set14_dataloader, urban100_dataloader, manga109_dataloader


class Dataset_PairedImage(Dataset):

    def __init__(self, scale=2, gt_size=128, geometric_augs=True, repeat=1):
        super(Dataset_PairedImage, self).__init__()
        self.scale = scale

        root = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))

        self.hr_p = os.path.join(root, f'Datasets{os.sep}DF2K{os.sep}DF2K_HR_train')
        self.hr = _get_paths_from_images(self.hr_p)

        self.lr_p = os.path.join(root, f'Datasets{os.sep}DF2K{os.sep}DF2K_HR_train_x{self.scale}m')
        self.lr = _get_paths_from_images(self.lr_p)

        self.gt_size = gt_size
        self.geometric_augs = geometric_augs
        self.repeat = repeat

    def __getitem__(self, index):
        index = self._get_index(index)
        gt_path = self.hr[index]
        lr_path = self.lr[index]

        assert os.path.basename(gt_path) == os.path.basename(lr_path), f'gt_path:{os.path.basename(gt_path)}, lr_path:{os.path.basename(lr_path)}'
        img_gt = np.array(Image.open(gt_path), dtype=np.uint8)
        img_lq = np.array(Image.open(lr_path), dtype=np.uint8)
        img_gt, img_lq = paired_random_crop_(img_gt,img_lq,self.gt_size,self.scale)

        if self.geometric_augs:
            img_gt, img_lq = random_augmentation(img_gt, img_lq)

        img_gt, img_lq = self.np2tensor(img_gt), self.np2tensor(img_lq)
        return img_lq, img_gt

    def __len__(self):
        return len(self.hr) * self.repeat

    def _get_index(self, idx):
        return idx % len(self.hr)

    def np2tensor(self, imgs):
        return torch.from_numpy(imgs.astype(np.float32) / 255.).float().permute(2,0,1)

    def tensor2np(self, imgs):
        imgs_np = np.uint8((imgs.data.cpu().numpy().squeeze(0).transpose(1,2,0).astype(np.float32).clip(0,1) * 255.).round())
        return imgs_np


class Benchmark(Dataset):
    def __init__(self, dataset='Set5', scale=2):
        super(Benchmark, self).__init__()
        assert dataset in ('Set5','Set14','Urban100', 'Manga109')
        self.scale = scale

        root = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))

        self.hr_p = os.path.join(root, f'Datasets/Benchmark/{dataset}/GTmod12')
        self.hr = _get_paths_from_images(self.hr_p)

        self.lr_p = os.path.join(root, f'Datasets/Benchmark/{dataset}/LRbicx{self.scale}')

    def __len__(self):
        return len(self.hr)

    def __getitem__(self, index):
        gt_path = self.hr[index]
        hr_basename = os.path.basename(gt_path)
        lr_path = os.path.join(self.lr_p, hr_basename)

        img_gt = np.array(Image.open(gt_path), dtype=np.uint8)
        img_lq = np.array(Image.open(lr_path), dtype=np.uint8)

        return self.np2tensor(img_lq), self.np2tensor(img_gt)

    def np2tensor(self, imgs):
        return torch.from_numpy(imgs.astype(np.float32) / 255.).float().permute(2,0,1)
