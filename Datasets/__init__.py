import numpy as np

from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as torchvision_datasets

from utils import *


def CIFAR10SSL(root: str, N1, M1, val_size,\
               rho_l, rho_u, inc_lb, seed=0, imb_type="exp",\
               transform_l=None, transform_u = None, transform_val=None,\
               target_transform = None, download: bool = False):
    def mutate_dataset(dataset, idx):
        dataset.data = dataset.data[idx]
        dataset.targets = np.array(dataset.targets)[idx].tolist()
        return dataset
    """
    Args:
        csv_file (string): Path to the csv file with annotations.
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """
    dataset_l = torchvision_datasets.CIFAR10(root=root, train=True, transform=transform_l,
                                             target_transform=target_transform, download=download)
    dataset_u = torchvision_datasets.CIFAR10(root=root, train=True, transform=transform_u,
                                             target_transform=None, download=download)
    dataset_val = torchvision_datasets.CIFAR10(root=root, train=True, transform=transform_val,
                                               target_transform=target_transform, download=download)
    targets = dataset_l.targets
    lb_idx, ulb_idx, val_idx = SSLSplitCIFAR(targets=targets, num_classes=10, N1=N1,\
                                             M1=M1, val_size=val_size, rho_l=rho_l, rho_u=rho_u,
                                             inc_lb=inc_lb, seed=seed, imb_type=imb_type)
    dataset_l = mutate_dataset(dataset_l, lb_idx)
    dataset_u = mutate_dataset(dataset_u, ulb_idx)
    dataset_val = mutate_dataset(dataset_val, val_idx)
    return dataset_l, dataset_u, dataset_val
    

def CIFAR100SSL(root: str, N1, M1, val_size,\
               rho_l, rho_u, inc_lb, seed=0, imb_type="exp",\
               transform_l=None, transform_u = None, transform_val=None,\
               target_transform = None, download: bool = False):
    def mutate_dataset(dataset, idx):
        dataset.data = dataset.data[idx]
        dataset.targets = np.array(dataset.targets)[idx].tolist()
        return dataset
    """
    Args:
        csv_file (string): Path to the csv file with annotations.
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """
    dataset_l = torchvision_datasets.CIFAR100(root=root, train=True, transform=transform_l,
                                             target_transform=target_transform, download=download)
    dataset_u = torchvision_datasets.CIFAR100(root=root, train=True, transform=transform_u,
                                             target_transform=None, download=download)
    dataset_val = torchvision_datasets.CIFAR100(root=root, train=True, transform=transform_val,
                                               target_transform=target_transform, download=download)
    targets = dataset_l.targets
    lb_idx, ulb_idx, val_idx = SSLSplitCIFAR(targets=targets, num_classes=100, N1=N1,\
                                             M1=M1, val_size=val_size, rho_l=rho_l, rho_u=rho_u,
                                             inc_lb=inc_lb, seed=seed, imb_type=imb_type)
    dataset_l = mutate_dataset(dataset_l, lb_idx)
    dataset_u = mutate_dataset(dataset_u, ulb_idx)
    dataset_val = mutate_dataset(dataset_val, val_idx)
    return dataset_l, dataset_u, dataset_val


