# Copyright 2024 Ilana Sebag, Muni Sreenivas Pydi, Jean-Yves Franceschi, Alain Rakotomamonjy, Mike Gartrell, Jamal Atif, Alexandre Allauzen

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import torch
import torch.nn.functional as F

from tqdm import trange

import matplotlib.pyplot as plt
import numpy as np

import scipy as sp
from scipy import stats

from DPSWflow.flow.utils import interp1d_

from DPSWflow.gaussian_moments import *
import math
import sys

import numpy as np
import scipy.integrate as integrate
import scipy.stats
from mpmath import mp

from itertools import cycle 

def sensitivity_bound(n_projs, dim, delta):
    if n_projs > 30:
        icdf = sp.stats.norm.ppf(1-delta/2)
        bound = math.sqrt(n_projs / dim + (icdf/dim) * math.sqrt((2*n_projs * (dim-1))/(dim+2)))
    else:
        bound = math.sqrt( n_projs/dim + (2/3)*math.log(1/delta)  + (2/dim) * math.sqrt(  n_projs * ((dim-1)/(dim+2) ) *  math.log(1/delta) )  )
    return 2*bound


def get_sigma(n_projs, dim, epsilon, delta):
    sensitivity = sensitivity_bound(n_projs, dim, delta)
    c = math.sqrt(2*math.log(1.25 / (delta/2)) + 1e-150)
    sigma = c * sensitivity / epsilon
    return sigma


def plot(step, xk, autoencoder, data_shape, device, exp_path):

    gen_imgs = autoencoder.decode(xk[:25].to(device)).cpu()
    if data_shape[0] == 1:
        gen_imgs = torch.cat([gen_imgs, gen_imgs, gen_imgs], dim=1)

    r, c = 5, 5
    cpt = 0
    fig,ax = plt.subplots(r,c)
    for i in range(r):
        for j in range(c):
            ax[i,j].imshow(np.transpose(gen_imgs[cpt],(1,2,0)))
            ax[i,j].axis("off")
            cpt += 1

    fig.set_size_inches(6, 6)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_path, f'img_{step}'))
    plt.close()


@torch.inference_mode()
def swf(train_set2, n, num_steps,num_steps_resampling, resample_thetas, batch_size, image_width, num_projs, num_sub_projs, num_percentiles, sigma_noise, dt, reg, epsilon, delta, target, autoencoder, val_freq, data_shape, device, exp_path):
    print(target.size(0))
    d = target.size(1)

    percentiles = torch.linspace(0, 1, num_percentiles)

    # particle processing
    x0 = torch.randn(n, d)
    xk = x0.clone()

    # flow init
    L = [x0.cpu().clone()]


    if resample_thetas == False:

        eps = 0

        sigma = get_sigma(num_projs, d, epsilon, delta)
        print(sigma)

        # thetas processing
        manythetas = torch.randn(num_projs, d)
        manythetas = F.normalize(manythetas, p=2, dim=1)
        unif = torch.ones(manythetas.shape[0])
        idx = unif.multinomial(num_sub_projs, replacement=True)

        pbar = trange(num_steps)
        for k in pbar:
                
            theta = manythetas[idx]
            
            target_proj = (target@theta.T).T
            target_proj_noised = torch.normal(target_proj, sigma)
            target_quantiles = torch.quantile(target_proj_noised, percentiles, dim=1).T

            xk_proj_noised = torch.normal((xk@theta.T).T, sigma)

            xk_quantiles = torch.quantile(xk_proj_noised, percentiles, dim=1).T
            cdf_xk = interp1d_(xk_quantiles, percentiles, xk_proj_noised)

            q = interp1d_(percentiles.repeat(num_sub_projs,1), target_quantiles, cdf_xk)

            nabla = ((xk_proj_noised-q)[:,None,:]*theta[:,:,None]).mean(dim=0).T

            zk = torch.randn_like(xk)
            xk = xk - dt * nabla + math.sqrt(2*dt*reg) * zk
            L.append(xk.clone())

            if k % 50 ==0 or k == num_steps-1 or k == num_steps:
                plot(k, xk, autoencoder, data_shape, device, exp_path)

    
    elif resample_thetas == True:

        sigma = sigma_noise
        sensitivity = sensitivity_bound(num_projs, d, delta)
        c = math.sqrt(2*math.log(1.25 / (delta/2)) + 1e-150)

        num_steps = int(num_steps_resampling * len(train_set2) / batch_size)
        pbar = trange(num_steps)

        max_lmbd = 32
        lmbds = range(1, max_lmbd + 1)
        log_moments = []
        q_batch=batch_size/len(train_set2)
        T=num_steps
        log_moment = 0
        for lmbd in lmbds:
                log_moment += compute_log_moment(q_batch, sigma, T, lmbd)
                log_moments.append((lmbd, log_moment))
        eps, delta = get_privacy_spent(log_moments, target_delta=delta)
        print('eps = ', eps)


        train_sampler = torch.utils.data.DataLoader(train_set2, batch_size=batch_size, shuffle=True, drop_last=True)
        dataiter = iter(cycle(train_sampler))

        
        sigma = sigma*sensitivity 

        print('sigma = ', sigma )
        for k in pbar:
            
            target = autoencoder.encode(next(dataiter)[0].to(device))
            target = target.detach().cpu()

            # data processing
            theta = torch.randn(num_projs, d)
            theta = F.normalize(theta, p=2, dim=1) 
            target_proj = (target@theta.T).T
            target_proj_noised = torch.normal(target_proj, sigma)
            target_quantiles = torch.quantile(target_proj_noised, percentiles, dim=1).T

            xk_proj_noised = torch.normal((xk@theta.T).T, sigma)

            xk_quantiles = torch.quantile(xk_proj_noised, percentiles, dim=1).T
            cdf_xk = interp1d_(xk_quantiles, percentiles, xk_proj_noised)

            q = interp1d_(percentiles.repeat(num_projs,1), target_quantiles, cdf_xk)

            nabla = ((xk_proj_noised-q)[:,None,:]*theta[:,:,None]).mean(dim=0).T

            zk = torch.randn_like(xk)
            xk = xk - dt * nabla + math.sqrt(2*dt*reg) * zk
            L.append(xk.clone())


            if k % 30 ==0:
                plot(k, xk, autoencoder, data_shape, device, exp_path)

    plot(num_steps, xk, autoencoder, data_shape, device, exp_path)


    return L, eps, delta


