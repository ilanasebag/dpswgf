#Code from https://github.com/arakotom/dp_swd
# no known license, however this code has been used and is being used here for research purpose only 

import torch
import numpy as np
import os
import shutil
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import *

from model import generator
from model.args import create_args
from utils import DotDict, load_yaml
from data import data_factory
from model.trainlatentspace import *

from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor
from torchvision.datasets import CelebA, FashionMNIST, MNIST

from autoencoder.train import AutoEncoder

@torch.inference_mode()
def evaluate_fid(gen_particles, test_loader, device, opt):
    fid = FrechetInceptionDistance(normalize=True, sync_on_compute=False).to(device)
    gen_particles=torch.from_numpy(gen_particles)
    if gen_particles.shape[0] == 1:
        gen_particles = torch.cat([gen_particles, gen_particles, gen_particles], dim=1)
    gen_particles=gen_particles.to(device)
    print(gen_particles.shape)
    for particles in gen_particles.split(opt.optim.batch_size):
        fid.update(gen_particles, real=False)
    print('t1')
    for imgs, _ in test_loader:
        if imgs.size(1) == 1:
            imgs = torch.cat([imgs, imgs, imgs], dim=1)
        fid.update(imgs.to(device), real=True)
    print('t2')
    fid_value = fid.compute()
    return fid_value.item()



def main(opt):
    fidlist =[]
    for it in tqdm(range(10)):

        if opt.device is None:
            device = torch.device('cpu')
        else:
            device = torch.device(f'cuda:{opt.device}')
            assert cudnn.is_available()
            cudnn.benchmark = True
        
        np.random.seed(np.random.randint(0, 10))
        torch.manual_seed(np.random.randint(0, 10))
        train_set1, train_set2, val_set, test_set, data_shape = data_factory(opt.data, opt.data_path)

        n = len(train_set2)
        test_loader = DataLoader(test_set, batch_size=opt.optim.batch_size, shuffle=True,drop_last=True, num_workers=opt.num_workers,
                                    pin_memory=True)
        train2_loader = DataLoader(train_set2, batch_size=opt.optim.batch_size,shuffle=True, drop_last=True, num_workers=opt.num_workers,
                                    pin_memory=True)
        
        dataiter = iter(cycle(train2_loader))
        autoencoder = AutoEncoder(opt, data_shape)
        autoencoder.to(device)
        checkpoint = torch.load(os.path.join(opt.exp_path, 'model.pt'), map_location=device)
        autoencoder.load_state_dict(checkpoint['model_state_dict'])
        autoencoder.eval()
       
        g = generative_model(n,autoencoder, dataiter, opt.data.num_channels, train2_loader, opt.optim.iters,opt.optim.num_projections,  opt.generative_model.sigma_noise, opt.generative_model.delta, data_shape, opt.optim.batch_size, opt.autoencoder.latent_size, opt.generative_model.image_width,opt.optim.lr, opt.optim.device, opt.exp_path, model_name='test_experiment')

        eps, delta = g.train()
        gen_particles = g.generate_images()
        fidlist.append(evaluate_fid(gen_particles, test_loader, device, opt))

    print(f'FID: {evaluate_fid(gen_particles, test_loader, device, opt)}')
    np.savetxt(os.path.join(opt.exp_path, 'FID') , np.array(fidlist))

    print(eps, delta)
    np.savetxt(os.path.join(opt.exp_path, 'eps_delta') , np.array([eps, delta]))
    

    return

if __name__ == '__main__':
    
    # Arguments
    p = create_args()
    opt = DotDict(vars(p.parse_args()))
    config = load_yaml(opt.config)
    opt.update(config)
    shutil.copy(opt.config, opt.exp_path)

    # Main
    main(opt)
