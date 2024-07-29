#Code from https://github.com/arakotom/dp_swd
# no known license, however this code has been used and is being used here for research purpose only 

import os
import errno
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable

from model.generator import Generator
from distrib_distance import *
from tqdm import *

from gaussian_moments import *
import math
import sys


import numpy as np
import scipy.integrate as integrate
import scipy.stats
from mpmath import mp

import scipy as sp
from scipy import stats


class generative_model:
    def __init__(self,n, num_channels, data, iters, num_projections, sigma_noise, delta, data_shape, batch_size, latent_dim, image_width,learning_rate,device,  exp_path, model_name='test_experiment'):
        self.image_width = image_width
        self.num_channels = num_channels
        self.image_size = self.num_channels * (self.image_width**2)
        self.base_dir = os.path.join(exp_path, f'{model_name}')
        self.data=data
        self.iters = iters
        self.data_shape = data_shape
        self.n = n 
        self.batch_size=batch_size
        self.latent_dim=latent_dim
        self.device = device
        self.generator = Generator(self.latent_dim, self.data_shape)
        self.generator.to(device)
        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=learning_rate)
        self.sigma_noise = sigma_noise
        self.num_projections=num_projections
        self.delta=delta

    def dp_sw_loss(self, true_distribution, generated_distribution):
        device = self.device
        image_width= self.image_width*self.image_width
        sensitivity = self.sensitivity_bound()
        loss  = sliced_wasserstein_distance_diff_priv(true_distribution,generated_distribution, self.sigma_noise* sensitivity, self.num_projections, p=2, device=device,sigma_proj=1)   #change to p=2
        return loss

    def sensitivity_bound(self):
        dim = self.image_size
        if self.num_projections > 30:
            icdf = sp.stats.norm.ppf(1-self.delta/2)
            bound = math.sqrt(self.num_projections / dim + (icdf/dim) * math.sqrt((2*self.num_projections * (dim-1))/(dim+2)))
        else:
            bound = math.sqrt( self.num_projections/dim + (2/3)*math.log(1/self.delta)  + (2/dim) * math.sqrt(  self.num_projections * ((dim-1)/(dim+2) ) *  math.log(1/self.delta) )  )
        return 2*bound

    def train(self):

        device = self.device
        sensitivity = self.sensitivity_bound()
        max_lmbd = 30
        lmbds = range(1, max_lmbd + 1)
        log_moments = []
        q_batch=self.batch_size/self.n
        T= self.iters * self.n / self.batch_size
        print('T=', T)
        print ('q=', q_batch)
        print('sensitivity=', sensitivity)
        log_moment = 0
        sigma=self.sigma_noise
        for lmbd in lmbds:
                log_moment += compute_log_moment(q_batch, sigma, T, lmbd)
                log_moments.append((lmbd, log_moment))
        eps, delta = get_privacy_spent(log_moments, target_delta=self.delta)
        print('eps=',eps)
        print('sigma=', self.sigma_noise* sensitivity)

        for iteration in tqdm(range(self.iters)):
            for i, (images, _) in enumerate(self.data):

                images = images.view(self.batch_size, -1)
                x = images.to(device)
                z = torch.randn(self.batch_size, self.latent_dim)
                z = z.to(device)       
                x_hat = self.generator(z) 
                xhat=x_hat.view(self.batch_size, -1)
                generator_loss = self.dp_sw_loss(x.view(-1, self.image_width * self.image_width*self.num_channels), xhat)
                self.optimizer_g.zero_grad()
                generator_loss.backward()
                self.optimizer_g.step()

            if iteration % 1 ==0:
                print("Loss after iteration {}: {}".format(iteration, generator_loss.item()))

                gen_imgs = self.generator(z) 
                gen_imgs = gen_imgs.cpu()
                gen_imgs=gen_imgs.detach().numpy()
    
                if self.num_channels == 3:
                    gen_imgs_np = np.transpose(gen_imgs, (0, 2, 3, 1))

                    for i in range(9):
                        plt.subplot(330 + 1 + i)
                        plt.imshow(gen_imgs_np[i])
                        plt.tight_layout()
                        plt.savefig(self.base_dir + '/Iteration_{}.png'.format(iteration), bbox_inches='tight')
                        plt.close()
                
                elif self.num_channels == 1:
                    for i in range(9):
                        # define subplot
                        plt.subplot(330 + 1 + i)
                        # plot raw pixel data
                        plt.imshow(gen_imgs.reshape((self.batch_size,self.image_width,self.image_width))[i], cmap=plt.get_cmap('gray'))
                        plt.tight_layout()
                        plt.savefig(self.base_dir + '/Iteration_{}.png'.format(iteration), bbox_inches='tight')
                        plt.close()

            if iteration == (self.iters-1):
                torch.save({
                    'generator_state_dict': self.generator.state_dict(),
                    'optimizer_g_state_dict': self.optimizer_g.state_dict(),
                }, self.base_dir + '/checkpoint.pth')

        return eps, delta

    def generate_images(self):
        device = self.device
        checkpoint = torch.load(self.base_dir + '/checkpoint.pth')
        self.generator.load_state_dict(checkpoint['generator_state_dict'])

        z = torch.randn(self.batch_size, self.latent_dim) 
        z = z.to(device)
        gen_imgs_ = self.generator(z)
        gen_imgs = gen_imgs_.cpu()
        if self.data_shape[0] == 1:
            gen_imgs = torch.cat([gen_imgs, gen_imgs, gen_imgs], dim=1)
        gen_imgs=gen_imgs.detach().numpy()
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
        plt.savefig(self.base_dir + '/Samples.png', bbox_inches='tight')
        plt.close()
        return gen_imgs_