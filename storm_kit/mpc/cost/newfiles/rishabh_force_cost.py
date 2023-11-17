#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.#
import torch
import torch.nn as nn
# import torch.nn.functional as F
from .gaussian_projection import GaussianProjection
import numpy as np
import time

def get_contact_stiffness_matrix(n_ci):
    # TODO: Change k_default to be a parameter
    k_default = 1000.
    n_ci = np.nan_to_num(n_ci)
    # TODO: Fix this operation
    K_ci = np.outer(n_ci, n_ci)
    # K_ci = np.eye(3)
    K_ci = k_default * K_ci
    return K_ci

class ForceThresholdCost(nn.Module):
    def __init__(self, tensor_args={'device':torch.device('cpu'), 'dtype':torch.float64},
                 bounds=[], weight=1.0, gaussian_params={}, bound_thresh=0.1):
        super(ForceThresholdCost, self).__init__()
        self.tensor_args = tensor_args
        self.weight = torch.as_tensor(weight, **tensor_args)
        self.proj_gaussian = GaussianProjection(gaussian_params=gaussian_params)
        # TODO: Change default_k to be a parameter
        default_k = 1000.
        self.k_c = torch.eye(3, **tensor_args).fill_(default_k).unsqueeze(0)
        # TODO: Add variables for force threshold

    def forward(self, start_state, state_batch, contact_info):
        inp_device = state_batch.device

        # bound_mask = torch.logical_and(state_batch < self.bounds[:,1],
        #                                state_batch > self.bounds[:,0])
        # print ("contact_info: ", contact_info)
        if len(contact_info['normal']) == 0:
            return torch.zeros((state_batch.shape[0], state_batch.shape[1]), **self.tensor_args)
        if contact_info['force'] < 2.:
            return torch.zeros((state_batch.shape[0], state_batch.shape[1]), **self.tensor_args)

        start_state = start_state[:, :7]
        state_batch = state_batch[:, :, :7]
        state_batch = torch.as_tensor(state_batch - start_state, **self.tensor_args)
        cost = torch.square(state_batch)
        # normals = contact_info['normal']
        # jacobian = contact_info['jac']
        normals = torch.as_tensor(contact_info['normal'], **self.tensor_args)
        normals = normals.reshape(normals.shape[0], 1, -1) # (n_ci, 1, 3)
        jacobian = torch.as_tensor(contact_info['jac'], **self.tensor_args) # (n_ci, 6, 7)
        deltas = torch.empty((state_batch.shape[0], state_batch.shape[1], len(normals)), **self.tensor_args)
        # deltas = np.empty((state_batch.shape[0], state_batch.shape[1], len(normals)))
        t = time.time()
        out = torch.matmul(normals, self.k_c)
        out = torch.matmul(out, jacobian)
        for i in range(state_batch.shape[0]):
            for j in range(state_batch.shape[1]):
                delta = torch.matmul(out, state_batch[i,j,:].reshape(-1,1))
                deltas[i, j] = delta.reshape(-1)
        #         for k in range(len(normals)):
        #             out = torch.matmul(jacobian[k], diff[i,j])
                    # k_ci = get_contact_stiffness_matrix(n_ci)
                    # out = n_ci @ k_ci @  jacobian[k] @ (diff[i, j]).reshape(-1, 1)
                    # deltas[i, j, k] = out[0]
        # for i in range(state_batch.shape[0]):
        #     for j in range(state_batch.shape[1]):
        #         for k in range(len(normals)):
        #             n_ci = np.reshape(normals[k], (1, -1))
        #             k_ci = torch.as_tensor(get_contact_stiffness_matrix(n_ci), **self.tensor_args)
        #             n_ci = torch.as_tensor(np.reshape(normals[k], (1, -1)), **self.tensor_args)
        #             out = n_ci @ k_ci @  torch.as_tensor(jacobian[k], **self.tensor_args) @ (diff[i, j]).reshape(-1, 1)
        #             deltas[i, j, k] = out[0]
        out = out.squeeze(1)
        # print ("Out shape: ", out.shape)
        # deltas2 = torch.matmul(out, diff.T).T.reshape(diff.shape[0], diff.shape[1], -1)
        # # print ("device: ", deltas.device, " device2: ", deltas2.device)
        # print ("Diff: ", deltas2 - deltas)
        cost = torch.zeros((state_batch.shape[0], state_batch.shape[1]), **self.tensor_args).double()
        forces = torch.as_tensor(contact_info['force'], **self.tensor_args)
        print ("Avg force: ", torch.mean(forces, dim=0))
        fi = forces.reshape(1, 1, -1) + deltas
        fi = torch.sum(fi, dim=2)
        # print ("FI shape: ", fi.shape)
        cost = torch.where(fi > 3, 10000., cost)
        # cost = torch.where(fi < 5, 10000., cost)
        # cost = torch.sum(cost, dim=2)
        t = time.time() - t
        # print ("Time: ", t, " Size: ", out.size())
        print ("force cost: ", cost)
        return cost.to(inp_device)

    # def compute_rollout_forces(joint_config, diff, force_normals, contact_jacobians):
    #     """
    #     Compute the forces for the rollout
    #     """
    #     # TODO: Should be performed for the entire rollout
    #     deltas = []
    #     for i, j_ci in enumerate(contact_jacobians):
    #         n_ci = np.reshape(force_normals[i], (1, -1))
    #         k_ci = get_contact_stiffness_matrix(n_ci)
    #         deltas.append(n_ci @ k_ci @ j_ci @ (q_des - joint_config).reshape(-1, 1))
    #     return deltas