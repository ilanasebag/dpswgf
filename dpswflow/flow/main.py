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

import os
import shutil
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
import numpy as np
import matplotlib.pyplot as plt
from DPSWflow.autoencoder.train import AutoEncoder
from DPSWflow.flow.args import create_args
from DPSWflow.flow.swf_moment import swf
from DPSWflow.utils import DotDict, load_yaml
from DPSWflow.data import data_factory
from tqdm import * 


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


def generate_images(it, gen_particles, batch_size, autoencoder, device, exp_path,opt):
    decoded_particles = autoencoder.decode(gen_particles.to(device))
    if decoded_particles.size(1) == 1:
        decoded_particles = torch.cat([decoded_particles, decoded_particles, decoded_particles], dim=1)
    gen_imgs = decoded_particles.cpu()
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
    plt.savefig(os.path.join(exp_path, 'samples_%s.png'%it))
    plt.close()
    return decoded_particles



@torch.inference_mode()
def evaluate_fid(gen_particles, test_loader, autoencoder, device, opt):
    fid = FrechetInceptionDistance(normalize=True, sync_on_compute=False).to(device)
    for particles in gen_particles.split(opt.optim.batch_size):
        decoded_particles = autoencoder.decode(particles.to(device))
        if decoded_particles.size(1) == 1:
            decoded_particles = torch.cat([decoded_particles, decoded_particles, decoded_particles], dim=1)
        fid.update(decoded_particles, real=False)
    for imgs, _ in test_loader:
        if imgs.size(1) == 1:
            imgs = torch.cat([imgs, imgs, imgs], dim=1)
        fid.update(imgs.to(device), real=True)
    fid_value = fid.compute()
    return fid_value.item()


def main(opt):
    
    fidlist =[]

    for it in tqdm(range(1)):

        if opt.device is None:
            device = torch.device('cpu')
        else:
            device = torch.device(f'cuda:{opt.device}')
            assert cudnn.is_available()
     
        train_set1, train_set2, vasl_set, test_set, data_shape = data_factory(opt.data, opt.data_path)

        n = len(test_set)
        test_loader = DataLoader(test_set, batch_size=opt.optim.batch_size, num_workers=opt.num_workers,
                                pin_memory=True)

        target = torch.load(os.path.join(opt.exp_path, 'train_set_encoding.pt'))

        autoencoder = AutoEncoder(opt, data_shape)
        autoencoder.to(device)
        checkpoint = torch.load(os.path.join(opt.exp_path, 'model.pt'), map_location=device)
        autoencoder.load_state_dict(checkpoint['model_state_dict'])
        autoencoder.eval()

        gen_particles, eps,delta =swf(train_set2, n, opt.flow.num_steps, opt.flow.num_steps_resampling, opt.flow.resample_thetas,  opt.optim.batch_size, opt.data.image_size, opt.flow.num_projs, opt.flow.num_sub_projs, opt.flow.num_percentiles, opt.flow.sigma_noise, opt.flow.dt, opt.flow.reg,
                            opt.flow.epsilon, opt.flow.delta, target, autoencoder, opt.flow.val_freq, data_shape, device,
                            opt.exp_path)
        fidlist.append(evaluate_fid(gen_particles[-1], test_loader, autoencoder, device, opt))
    fidmean = np.mean(np.array(fidlist))
    fidvar = np.var(np.array(fidlist))
    epsdelta=[]
    epsdelta.append(eps)
    epsdelta.append(delta)
    np.savetxt(os.path.join(opt.exp_path, 'FID') , np.array(fidlist))
    np.savetxt(os.path.join(opt.exp_path, 'eps_delta') , np.array(epsdelta))

if __name__ == '__main__':
    # Arguments
    p = create_args()
    opt = DotDict(vars(p.parse_args()))
    config = load_yaml(opt.config)
    opt.update(config)
    shutil.copy(opt.config, opt.exp_path)

    # Main
    main(opt)


