# @File  :gaussian.py
# @Time  :2021/1/4
# @Desc  :
import numpy as np
import torch
import torch.nn as nn


class GaussianLayer(nn.Module):
    def __init__(self, input_dim, n_classes):
        super(GaussianLayer, self).__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        
        self.centers = nn.Parameter(
            0.5 * torch.randn(n_classes, input_dim))
        self.covs = nn.Parameter(0.2 + torch.tensor(np.random.exponential(
            scale=0.5, size=(n_classes, input_dim))))
        
    def forward(self, x):
        covs = self.covs.unsqueeze(0).expand(
            x.size(0), self.n_classes, self.input_dim)  # [bs, n_classes, dim]
        centers = self.centers.unsqueeze(0).expand(
            x.size(0), self.n_classes, self.input_dim)  # [bs, n_classes, dim]
        
        diff = x.unsqueeze(1).repeat(1, self.n_classes, 1) - centers
        z_log = -0.5 * torch.sum(
            torch.log(self.covs), dim=-1) - 0.5 * self.input_dim * np.log(
            2 * np.pi)
        exp_log = -0.5 * torch.sum(
            diff * (1 / (covs + np.finfo(np.float32).eps)) * diff, dim=-1)

        return torch.nn.functional.normalize(z_log + exp_log, p=2)
    
    def clip_convs(self):
        with torch.no_grad():
            self.covs.clamp_(np.finfo(np.float32).eps)
