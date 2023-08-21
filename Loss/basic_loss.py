import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        loss = torch.mean(torch.sqrt((x - y) ** 2 + (self.eps*self.eps)))
        return loss


class FFTLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FFTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.criterion = torch.nn.L1Loss(reduction=reduction)

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)

        pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
        target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)

        return self.loss_weight * self.criterion(pred_fft, target_fft)


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        """
        Total variation loss
        https://github.com/jxgu1016/Total_Variation_Loss.pytorch
        Args:
            tv_loss_weight (int):
        """
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]