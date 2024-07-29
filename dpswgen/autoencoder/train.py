import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm import *

from autoencoder.args import create_args
from autoencoder.encoder import decoder_factory, normalized_encoder_factory
from utils import DotDict, load_yaml
from data import data_factory


class AutoEncoder(nn.Module):
    def __init__(self, opt, data_shape):
        super().__init__()
        self.encode = normalized_encoder_factory(data_shape, opt.autoencoder.latent_size, opt.autoencoder.encoder)
        self.decode = decoder_factory(opt.autoencoder.latent_size, data_shape, opt.autoencoder.decoder)

    def encode_nograd(self, x):
        with torch.no_grad():
            return self.encode(x)

    def decode_nograd(self, x):
        with torch.no_grad():
            return self.decode(x)


def loss_fn_choice(name):
    if name == 'bce':
        return nn.BCELoss()
    if name == 'mse':
        return nn.MSELoss()
    raise ValueError(f'No loss named \'{name}\'')


def encode_datasets(opt, train_set, val_set, model, device, load):
    train_loader = DataLoader(train_set, batch_size=opt.optim.batch_size, num_workers=opt.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=opt.optim.batch_size, num_workers=opt.num_workers, pin_memory=True)
    if load:
        checkpoint = torch.load(os.path.join(opt.exp_path, 'model.pt'), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    encoded_train = []
    encoded_val = []
    for x, _ in tqdm(train_loader):
        target = model.encode_nograd(x.to(device)).cpu()
        encoded_train.append(target)
    for x, _ in tqdm(val_loader):
        target = model.encode_nograd(x.to(device)).cpu()
        encoded_val.append(target)
    encoded_train = torch.cat(encoded_train, dim=0)
    encoded_val = torch.cat(encoded_val, dim=0)
    model.train()

    torch.save(encoded_train, os.path.join(opt.exp_path, 'train_set_encoding.pt'))
    torch.save(encoded_val, os.path.join(opt.exp_path, 'val_set_encoding.pt'))


def main(opt):

    if opt.device is None:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{opt.device}')
        assert cudnn.is_available()
        cudnn.benchmark = True

    train_set1, train_set2, val_set, test_set, data_shape = data_factory(opt.data, opt.data_path)

    train1_loader = DataLoader(train_set1, batch_size=opt.optim.batch_size, shuffle=True, num_workers=opt.num_workers,
                               pin_memory=True, drop_last=True, persistent_workers=opt.num_workers > 0)
    train2_loader = DataLoader(train_set2, batch_size=opt.optim.batch_size, shuffle=True, num_workers=opt.num_workers,
                               pin_memory=True, drop_last=True, persistent_workers=opt.num_workers > 0)

    val_loader = DataLoader(val_set, batch_size=opt.optim.batch_size, shuffle=True, num_workers=opt.num_workers,
                            pin_memory=True, drop_last=True, persistent_workers=opt.num_workers > 0)

    model = AutoEncoder(opt, data_shape)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=opt.optim.lr, weight_decay=opt.optim.weight_decay)
    loss_fn = loss_fn_choice(opt.autoencoder.loss)

    step = 0

    finished = False or opt.load_only
    pb = tqdm(initial=step, total=opt.optim.num_steps, ncols=0) if not finished else None
    while not finished:
        train_loss = 0
        train_list = []
        internal_step = 0
        for batch, _ in train1_loader:
            if step >= opt.optim.num_steps:
                finished = True
                break
            x = batch.to(device)
            optimizer.zero_grad()
            x_ = model.decode(model.encode(x))
            loss = loss_fn(x_, x)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            step += 1
            internal_step += 1
            if step % opt.optim.val_freq == opt.optim.val_freq - 1:
                break

            pb.set_postfix(loss=train_loss / internal_step,refresh=False)
            pb.update()

        train_list.append(train_loss / internal_step)
        if step % opt.optim.val_freq == opt.optim.val_freq - 1 or finished:
            val_step = 0
            model.eval()
            with torch.no_grad():
                for batch, _ in val_loader:
                    x = batch.to(device)
                    x_ = model.decode(model.encode(x))
                    val_step += 1
                    if val_step >= opt.optim.val_iter:
                        break
            model.train()
            torch.save({'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'avg_loss': train_loss / internal_step,
                        'train_list': train_list},
                       os.path.join(opt.exp_path, 'model.pt'))

    encode_datasets(opt, train_set2, val_set, model, device, opt.load_only)


if __name__ == '__main__':
    # Arguments
    p = create_args()
    opt = DotDict(vars(p.parse_args()))
    config = load_yaml(opt.config)
    opt.update(config)
    shutil.copy(opt.config, opt.exp_path)

    # Main
    main(opt)
