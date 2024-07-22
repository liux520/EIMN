import torch
import os
import cv2
from tqdm import tqdm
import numpy as np
from natsort import os_sorted
from Metrics.psnr_ssim import _calculate_psnr_ssim_niqe


class AverageMeter():
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _get_paths_from_images(path, suffix=''):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname) and suffix in fname:
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return os_sorted(images)


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


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


def load(path, model, key='state_dict', delete_module=False):
    checkpoint = torch.load(path, map_location='cpu')
    checkpoint = checkpoint if len(checkpoint) > 10 else checkpoint[key]

    model_dict = model.state_dict()
    if delete_module:
        checkpoint = delete_state_module(checkpoint)
    overlap = {k: v for k, v in checkpoint.items() if k in model_dict}
    model_dict.update(overlap)
    print(f'{(len(overlap) * 1.0 / len(checkpoint) * 100):.4f}% weights is loaded!', end='\t')
    print(f'{(len(overlap) * 1.0 / len(model_dict) * 100):.4f}% params is init!')
    print(f'Drop Keys: {[k for k, v in checkpoint.items() if k not in model_dict]}')
    model.load_state_dict(model_dict)
    return model


def delete_state_module(weights):
    weights_dict = {}
    for k, v in weights.items():
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] = v
    return weights_dict


@torch.no_grad()
def test_on_custom_dataset(
        lr_path: str,
        hr_path: str,
        model,
        device,
        crop_border=0,
        test_y_channel=True,
        save=False,
        save_path='',
):
    if save:
        check_dir(save_path)

    lr_imgs = _get_paths_from_images(lr_path)
    hr_imgs = _get_paths_from_images(hr_path)

    model.to(device)
    model.eval()

    psnr, ssim = AverageMeter(), AverageMeter()

    for i, (lr_img, hr_img) in enumerate(zip(lr_imgs, hr_imgs)):
        base, ext = os.path.splitext(os.path.basename(lr_img))
        lr = cv2.imread(lr_img)[:, :, ::-1]
        hr = cv2.imread(hr_img)[:, :, ::-1]

        lr_tensor = uint2tensor(lr).to(device)
        hr_tensor = uint2tensor(hr).to(device)
        output = model(lr_tensor)

        psnr_temp, ssim_temp, _, batch = _calculate_psnr_ssim_niqe(output, hr_tensor, crop_border=crop_border,
                                                                   input_order='CHW', test_y_channel=test_y_channel,
                                                                   mean=(0, 0, 0), std=(1, 1, 1))
        psnr.update(psnr_temp, batch)
        ssim.update(ssim_temp, batch)

        print(f'Processing {i}: LR:{lr_img} | HR:{hr_img} | PSNR/SSIM:{psnr_temp:.4f}/{ssim_temp:.4f}')

        if save:
            output_copy = tensor2uint8(output)
            cv2.imwrite(os.path.join(save_path, f'{base}{ext}'), output_copy[:, :, ::-1])

    avg_psnr = psnr.avg
    avg_ssim = ssim.avg
    print(f'Avg PSNR:{avg_psnr} | Avg SSIM: {avg_ssim}')

    return avg_psnr, avg_ssim


def _model_dict_(model_name, scale):
    from Model.EIMN import EIMN_A, EIMN_L

    model_eimn_a = EIMN_A(scale=scale)
    model_eimn_l = EIMN_L(scale=scale)

    if model_name == 'EIMN_A_x2':
        model = load(r'../Weights/EIMN_A_x2.pth', model_eimn_a)
    elif model_name == 'EIMN_A_x3':
        model = load(r'../Weights/EIMN_A_x3.pth', model_eimn_a)
    elif model_name == 'EIMN_A_x4':
        model = load(r'../Weights/EIMN_A_x4.pth', model_eimn_a)
    elif model_name == 'EIMN_L_x2':
        model = load(r'../Weights/EIMN_L_x2.pth', model_eimn_l)
    elif model_name == 'EIMN_L_x3':
        model = load(r'../Weights/EIMN_L_x3.pth', model_eimn_l)
    elif model_name == 'EIMN_L_x4':
        model = load(r'../Weights/EIMN_L_x4.pth', model_eimn_l)
    return model


if __name__ == '__main__':
    """
    If you want to test the model on your own dataset, the simplest test code is provided here.
    You only need to modify the following:
    [1] scale: 2/3/4
    [2] model_name: EIMN_A_x2/EIMN_A_x3/EIMN_A_x4/EIMN_L_x2/EIMN_L_x3/EIMN_L_x4
    [3] lr_path/hr_path: LR-HR dir path of your own dataset. 
    """

    scale = 2
    model_name = 'EIMN_A_x2'
    model = _model_dict_(model_name, scale)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_on_custom_dataset(
        # Custom LR/HR images dir path
        lr_path=r'E:\Dataset\Restoration\SR\Benchmark\Set5\LRbicx2',
        hr_path=r'E:\Dataset\Restoration\SR\Benchmark\Set5\GTmod12',
        # Selected model
        model=model,
        device=device,
        # Test PSNR/SSIM configs
        crop_border=scale,
        test_y_channel=True,
        # Save output or not
        save=False,
        save_path=r''
    )

