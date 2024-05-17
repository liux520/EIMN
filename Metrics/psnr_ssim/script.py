import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from Metrics.psnr_ssim import calculate_psnr, calculate_ssim
from natsort import os_sorted


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


def test(denoised_path, gt_path, suffix_denoise='', suffix_gt=''):
    noise_imgs = _get_paths_from_images(denoised_path, suffix_denoise)
    gt_imgs = _get_paths_from_images(gt_path, suffix_gt)

    psnr_list, ssim_list = [], []

    for noise, gt in zip(noise_imgs, gt_imgs):
        noise_img = cv2.imread(noise)[:, :, ::-1]
        gt_img = cv2.imread(gt)[:, :, ::-1]

        # psnr = compare_psnr(noise_img, gt_img)
        # ssim = compare_ssim(noise_img, gt_img, data_range=255, gaussian_weights=True,
        #                     channel_axis=2, use_sample_covariance=False)

        psnr = calculate_psnr(noise_img, gt_img, crop_border=0, input_order='HWC', test_y_channel=True)
        ssim = calculate_ssim(noise_img, gt_img, crop_border=0, input_order='HWC', test_y_channel=True)

        psnr_list.append(psnr)
        ssim_list.append(ssim)
        print(f'{os.path.basename(noise)}: PSNR:{psnr} SSIM:{ssim}')

    mean_psnr, mean_ssim = np.mean(psnr_list), np.mean(ssim_list)
    print(f'Avg PSNR:{mean_psnr} Avg SSIM:{mean_ssim}')



if __name__ == '__main__':
    test(
        denoised_path=r'',
        gt_path=r'',
        suffix_denoise='',
        suffix_gt=''
    )
