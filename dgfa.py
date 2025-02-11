import torch
import torch.nn as nn
import scipy.stats as st
import torch.nn.functional as F
from utils.attack import Attack
import numpy as np
from torch.autograd import Variable as V
from utils.dct import *

def clip_by_tensor(t, t_min, t_max):
        """
        clip_by_tensor
        :param t: tensor
        :param t_min: min
        :param t_max: max
        :return: cliped tensor
        """
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result

def gkern(kernlen=15, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    kernel = kernel.astype(np.float32)
    gaussian_kernel = np.stack([kernel, kernel, kernel])  # 5*5*3
    gaussian_kernel = np.expand_dims(gaussian_kernel, 1)  # 1*5*5*3
    gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()  # tensor and cuda
    return gaussian_kernel

"""Input diversity: https://arxiv.org/abs/1803.06978"""
def DI(x, resize_rate=1.15, diversity_prob=0.5):
    assert resize_rate >= 1.0
    assert diversity_prob >= 0.0 and diversity_prob <= 1.0
    img_size = x.shape[-1]
    img_resize = int(img_size * resize_rate)
    rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
    rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
    h_rem = img_resize - rnd
    w_rem = img_resize - rnd
    pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
    pad_right = w_rem - pad_left
    padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)
    if img_size == 224:
        padded = F.interpolate(padded, size=[img_size, img_size], mode='bilinear', align_corners=False)
    ret = padded if torch.rand(1) < diversity_prob else x
    return ret


def DI_Resize(x, resize_rate=1.15, diversity_prob=0.5):
    assert resize_rate >= 1.0
    assert diversity_prob >= 0.0 and diversity_prob <= 1.0
    img_size = x.shape[-1]
    img_resize = int(img_size * resize_rate)
    rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
    rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
    h_rem = img_resize - rnd
    w_rem = img_resize - rnd
    pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
    pad_right = w_rem - pad_left
    padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)
    ret = padded if torch.rand(1) < diversity_prob else x
    ret = F.interpolate(ret, size=[img_size, img_size], mode='bilinear', align_corners=False)
    return ret

class DGFA(Attack):
    def __init__(self, model, eps=8/255,
                 alpha=2/255, steps=10,N=20,delta=0.5,rho=0.5):
        super().__init__("DGFA", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.supported_mode = ['default', 'targeted']
        self.T_kernel = gkern(7, 3)
        self.momentum = 1.0
        self.image_width = 299
        self.N = N
        self.sigma = 16
        self.delta = delta
        self.zeta=3.0
        self.rho = rho

    def pgn(self, x, model, labels):
        noise = 0
        for _ in range(self.N):
            gauss = torch.randn(x.size()[0], 3, self.image_width, self.image_width) * (self.sigma / 255)
            gauss = gauss.cuda()
            x_dct = dct_2d(x + gauss).cuda()
            mask = (torch.rand_like(x) * 2 * self.rho + 1 - self.rho).cuda()
            x_near = idct_2d(x_dct * mask)
            x_near = V(x_near, requires_grad = True)
            output_v3 = model(x_near)
            loss = F.cross_entropy(output_v3, labels)
            g1 = torch.autograd.grad(loss, x_near,
                                        retain_graph=False, create_graph=False)[0]
            x_star = x_near.detach() + self.alpha * (-g1)/torch.abs(g1).mean([1, 2, 3], keepdim=True)

            nes_x = x_star.detach()
            nes_x = V(nes_x, requires_grad = True)
            output_v3 = model(nes_x)
            loss = F.cross_entropy(output_v3, labels)
            g2 = torch.autograd.grad(loss, nes_x,
                                        retain_graph=False, create_graph=False)[0]

            noise += (1-self.delta)*g1 + self.delta*g2
        return noise

    def forward(self, images, labels,save_func,output_dir):
        self.image_width = images.shape[-1]
        x = images.clone()
        model = self.model
        images_min = clip_by_tensor(images - self.eps, 0.0, 1.0)
        images_max = clip_by_tensor(images + self.eps, 0.0, 1.0)
        grad = 0
        for i in range(self.steps):

            noise = self.pgn(x, model, labels)
            noise = (noise) / torch.abs(noise).mean([1, 2, 3], keepdim=True)
            noise = self.momentum * grad + noise
            grad = noise

            x = x + self.alpha * torch.sign(noise)
            x = clip_by_tensor(x, images_min, images_max)
            
        adv_img_np = x.detach().cpu().numpy()
        adv_img_np = np.transpose(adv_img_np, (0, 2, 3, 1)) * 255
        save_func(images=adv_img_np,output_dir=output_dir[:-1])
        return x.detach()
