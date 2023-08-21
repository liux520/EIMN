import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_gan_loss_D(criterion, d_pred_real, d_pred_fake):

    loss_real = criterion(d_pred_real, target_is_real=True, is_disc=True)
    loss_fake = criterion(d_pred_fake, target_is_real=False, is_disc=True)

    return (loss_real + loss_fake) / 2


def calculate_gan_loss_G(criterion, d_pred_fake):

    loss_real = criterion(d_pred_fake, target_is_real=True, is_disc=False)

    return loss_real


# def calculate_gan_loss_D(netD, criterion, real, fake):
#
#     d_pred_fake = netD(fake.detach())
#     d_pred_real = netD(real)
#
#     loss_real = criterion(d_pred_real, target_is_real=True, is_disc=True)
#     loss_fake = criterion(d_pred_fake, target_is_real=False, is_disc=True)
#
#     return (loss_real + loss_fake) / 2, d_pred_real, d_pred_fake
#
#
# def calculate_gan_loss_G(netD, criterion, real, fake):
#
#     d_pred_fake = netD(fake)
#     loss_real = criterion(d_pred_fake, target_is_real=True, is_disc=False)
#
#     return loss_real, d_pred_fake


class GANLoss(nn.Module):
    """Define GAN loss_script.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'wgan_softplus':
            self.loss = self._wgan_softplus_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        """wgan loss_script.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss_script.
        """
        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input, target):
        """wgan loss_script with soft plus. softplus is a smooth approximation to the
        ReLU function.

        In StyleGAN2, it is called:
            Logistic loss_script for discriminator;
            Non-saturating loss_script for generator.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss_script.
        """
        return F.softplus(-input).mean() if target else F.softplus(input).mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        if self.gan_type in ['wgan', 'wgan_softplus']:
            return target_is_real
        target_val = (self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss_script module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss_script for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss_script value.
        """
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight