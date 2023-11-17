import torch
import torch.nn as nn
# import torch.nn.functional as F
from .gaussian_projection import GaussianProjection
import numpy as np
import time

class JointLimitCost(nn.Module):
    def __init__(self, weight=None, device=torch.device('cpu'), float_dtype=torch.float32):
        super(JointLimitCost, self).__init__()
        self.device = device
        self.float_dtype = float_dtype
        self.tensor_args = {'device':device, 'dtype':float_dtype}
        self.weight = torch.as_tensor(weight, device=device, dtype=float_dtype)

        self.model = torch.load('./weights/3-128-ran-larm.pt')
        

    def forward(self, joint_batch):
        inp_device = joint_batch.device
        cost = self.model(joint_batch)
        return cost.to(inp_device)
