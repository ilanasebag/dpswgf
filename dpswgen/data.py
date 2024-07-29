
import random
import torch
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor
from torchvision.datasets import CelebA, FashionMNIST, MNIST

#DATA SPLIT WITH 2 TRAINING SETS:
#TRAIN1 = PUBLIC DATA AND TRAIN2 = PRIVATE DATA 

def split_train_val(dataset_len, ratio_val, rng):
    """
    Splits dataset indices into a training and validation partition using the given RNG seed.

    Useful for datasets which have no built-in validation set.
    """
    assert 0 < ratio_val < 1
    assert 0 < int((1 - ratio_val) * dataset_len) < dataset_len
    permutation = torch.randperm(dataset_len, generator=rng).tolist()
    train1 = permutation[:int((1 - ratio_val)/2 * dataset_len)]
    train2 = permutation[int((1 - ratio_val)/2 * dataset_len):int((1 - ratio_val) * dataset_len)]
    val = permutation[int((1 - ratio_val) * dataset_len):]
    return train1, train2, val

class ValMNIST(Dataset):
    """
    MNIST dataset, expanded to 32x32 by adding four layers of black pixels around the original images.

    The validation dataset is chosen by randomly removing elements of the original training dataset. The random choice
    is performed using the given seed to be shared with the constructor of the training dataset.
    """

    def __init__(self, data_path, split, ratio_val, seed=None):
        super().__init__()
        assert split in ['train1','train2', 'valid', 'test']
        transform = Compose([CenterCrop(32), ToTensor()])
        self.dataset = MNIST(data_path, train=split != 'test', transform=transform, download=True)
        # Split training dataset in training and validation sets
        if split != 'test':
            assert seed is not None and ratio_val is not None
            rng = torch.Generator()
            rng.manual_seed(seed)
            indices_train1, indices_train2, indices_val = split_train_val(len(self.dataset), ratio_val, rng)
            if split == 'train1':
                self.indices = indices_train1
            elif split == 'train2':
                self.indices = indices_train2
            else:
                self.indices = indices_val
        else:
            self.indices = range(len(self.dataset))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

class ValFashionMNIST(Dataset):
    """
    MNIST dataset, expanded to 32x32 by adding four layers of black pixels around the original images.

    The validation dataset is chosen by randomly removing elements of the original training dataset. The random choice
    is performed using the given seed to be shared with the constructor of the training dataset.
    """

    def __init__(self, data_path, split, ratio_val, seed=None):
        super().__init__()
        assert split in ['train1','train2', 'valid', 'test']
        transform = Compose([CenterCrop(32), ToTensor()])
        self.dataset = FashionMNIST(data_path, train=split != 'test', transform=transform, download=True)
        # Split training dataset in training and validation sets
        if split != 'test':
            assert seed is not None and ratio_val is not None
            rng = torch.Generator()
            rng.manual_seed(seed)
            indices_train1, indices_train2, indices_val = split_train_val(len(self.dataset), ratio_val, rng)
            if split == 'train1':
                self.indices = indices_train1
            elif split == 'train2':
                self.indices = indices_train2
            else:
                self.indices = indices_val
        else:
            self.indices = range(len(self.dataset))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


def data_factory(config, data_path):
    name = config.name
    if name == 'mnist':
        transform = Compose([CenterCrop(32), ToTensor()])
        seed = random.getrandbits(32)
        train_set1 = ValMNIST(data_path, 'train1', config.ratio_val, seed=seed)
        train_set2 = ValMNIST(data_path, 'train2', config.ratio_val, seed=seed)
        val_set = ValMNIST(data_path, 'valid', config.ratio_val, seed=seed)
        test_set = ValMNIST(data_path, 'test', config.ratio_val)
        shape = (1, 32, 32)
        return train_set1,train_set2, val_set, test_set, shape
    if name == 'fashion_mnist':
        transform = Compose([CenterCrop(32), ToTensor()])
        seed = random.getrandbits(32)
        train_set1 = ValFashionMNIST(data_path, 'train1', config.ratio_val, seed=seed)
        train_set2 = ValFashionMNIST(data_path, 'train2', config.ratio_val, seed=seed)
        val_set = ValFashionMNIST(data_path, 'valid', config.ratio_val, seed=seed)
        test_set = ValFashionMNIST(data_path, 'test', config.ratio_val)
        shape = (1, 32, 32)
        return train_set1,train_set2, val_set, test_set, shape

    if name == 'celeba':
        image_size =config.image_size 
        assert 0 < image_size <= 178
        transform = Compose([Resize(image_size), CenterCrop(image_size), ToTensor()])
        train_set = CelebA(data_path, split='train', target_type='attr', transform=transform, download=True)
        set_size1 = int(0.5 * len(train_set))
        set_size2 = len(train_set) - set_size1
        train_set1, train_set2 = torch.utils.data.random_split(train_set, [set_size1,set_size2])
        val_set = CelebA(data_path, split='valid', target_type='attr', transform=transform, download=True)
        test_set = CelebA(data_path, split='test', target_type='attr', transform=transform, download=True)
        shape = (3, image_size, image_size)
        return train_set1, train_set2, val_set, test_set, shape
    raise ValueError(f'No dataset named \'{name}\'')