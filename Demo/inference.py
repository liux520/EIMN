from Metrics.psnr_ssim import _calculate_psnr_ssim_niqe
import torch
import torch.nn as nn
from Utils.imresize import imresize
from Utils.msic import AverageMeter
import os
from PIL import Image
import cv2
from tqdm import tqdm
import random
import numpy as np
import torchvision.transforms as tr
from torch.utils.data import Dataset, DataLoader
from Data.data_utils import _get_paths_from_images


class Test_Dataset(Dataset):
    def __init__(self, hr_path='', scale=2):
        super(Dataset, self).__init__()

        lr_path = os.path.join(os.path.abspath(os.path.join(hr_path, os.path.pardir)), f'LRbicx{scale}')

        self.hr_imgs = _get_paths_from_images(hr_path)
        self.lr_imgs = _get_paths_from_images(lr_path)

    def __len__(self):
        return len(self.hr_imgs)

    def __getitem__(self, item):
        hr_p = self.hr_imgs[item]
        lr_p = self.lr_imgs[item]

        hr = np.array(Image.open(hr_p)).astype(np.float32) / 255.
        lr = np.array(Image.open(lr_p)).astype(np.float32) / 255.
        return self.np2tensor(hr), self.np2tensor(lr), os.path.basename(hr_p)

    def np2tensor(self, img):
        return torch.from_numpy(img.transpose(2, 0, 1)).float()


@torch.no_grad()
def Test(dataloader, model, device, record=False, save=False, dataset=None, scale=2,
         save_path='', record_path=''):
    if save:
        save_path = os.path.join(save_path, f'{dataset}{os.sep}x{scale}')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    psnr, ssim = AverageMeter(), AverageMeter()

    for i, (hr, lr, basename) in enumerate(tqdm(dataloader)):
        hr, lr = hr.to(device), lr.to(device)
        output = model(lr)

        psnr_temp, ssim_temp, _, batch = _calculate_psnr_ssim_niqe(output, hr, crop_border=scale,
                                                                           input_order='CHW',
                                                                           test_y_channel=True, mean=(0, 0, 0),
                                                                           std=(1, 1, 1))
        psnr.update(psnr_temp, batch)
        ssim.update(ssim_temp, batch)

        if save:
            output_copy = output.data.cpu().numpy().squeeze(0).transpose(1, 2, 0).astype(np.float32)
            output_copy = np.uint8((output_copy.clip(0, 1) * 255.).round())
            cv2.imwrite(os.path.join(save_path, basename[0]), output_copy[:,:,::-1])

    avg_psnr = psnr.avg
    avg_ssim = ssim.avg

    if record:
        with open(f'{record_path}{os.sep}record.txt', 'a+') as f:
            f.write(f'Scale:{scale} Dataset:{dataset} Avg_psnr:{avg_psnr} Avg_ssim:{avg_ssim}\n')

    return avg_psnr, avg_ssim


if __name__ == '__main__':

    # ------------ path ------------#
    root = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))
    set5_hr = os.path.join(root, rf'Datasets{os.sep}Benchmark{os.sep}Set5{os.sep}GTmod12')
    set14_hr = os.path.join(root, rf'Datasets{os.sep}Benchmark{os.sep}Set14{os.sep}GTmod12')
    urban_hr = os.path.join(root, rf'Datasets{os.sep}Benchmark{os.sep}urban100{os.sep}GTmod12')
    manga_hr = os.path.join(root, rf'Datasets{os.sep}Benchmark{os.sep}manga109{os.sep}GTmod12')
    bsds_hr = os.path.join(root, rf'Datasets{os.sep}Benchmark{os.sep}BSDS100{os.sep}GTmod12')

    model_name = 'EIMN_A'  # 'EIMN_L | EIMN_A
    scale = 4

    save = False
    save_path = rf''
    record = False
    record_path = rf''

    # ----------- device ---------- #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------- model ----------- #
    from Model.model_import import model_import

    model = model_import(model_name, scale=scale, load_weights=True).to(device)
    # model = nn.DataParallel(model, device_ids=[0])
    model.eval()
    print('--- Model load successfully! ---')

    # ---------- Dataloader ------- #
    set5_dataloader = DataLoader(Test_Dataset(set5_hr, scale=scale), batch_size=1)
    set14_dataloader = DataLoader(Test_Dataset(set14_hr, scale=scale), batch_size=1)
    urban_dataloader = DataLoader(Test_Dataset(urban_hr, scale=scale), batch_size=1)
    manga_dataloader = DataLoader(Test_Dataset(manga_hr, scale=scale), batch_size=1)
    bsds_dataloader = DataLoader(Test_Dataset(bsds_hr, scale=scale), batch_size=1)

    print('--- Data init successfully! ---')

    # ------------ test ----------- #
    psnr, ssim = Test(set5_dataloader, model, device, dataset='Set5', scale=scale,
                      save=save, record=record, save_path=save_path, record_path=record_path)
    print(f'\nScale:{scale} Set5: psnr:{psnr}, ssim:{ssim}')

    psnr, ssim = Test(set14_dataloader, model, device, dataset='Set14', scale=scale,
                      save=save, record=record, save_path=save_path, record_path=record_path)
    print(f'\nScale:{scale} Set14: psnr:{psnr}, ssim:{ssim}')

    psnr, ssim = Test(urban_dataloader, model, device, dataset='Urban100', scale=scale,
                      save=save, record=record, save_path=save_path, record_path=record_path)
    print(f'\nScale:{scale} Urban100: psnr:{psnr}, ssim:{ssim}')

    psnr, ssim = Test(manga_dataloader, model, device, dataset='Manga109', scale=scale,
                      save=save, record=record, save_path=save_path, record_path=record_path)
    print(f'\nScale:{scale} Manga: psnr:{psnr}, ssim:{ssim}')

    psnr, ssim = Test(bsds_dataloader, model, device, dataset='BSDS100', scale=scale,
                      save=save, record=record, save_path=save_path, record_path=record_path)
    print(f'\nScale:{scale} BSDS: psnr:{psnr}, ssim:{ssim}')
