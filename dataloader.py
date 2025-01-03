# Copyright (c) 2016, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
import random
import numpy as np


class DataLoaderWrapper:
    def __init__(self, dataset, opt, split):
        self.manual_seed = opt.manualSeed
        self.nCrops = 10 if split == 'val' and opt.tenCrop else 1
        self.batch_size = opt.batchSize // self.nCrops
        self.cpu_type = self.get_cpu_type(opt.tensorType)

        # Set seeds
        torch.manual_seed(self.manual_seed)
        random.seed(self.manual_seed)
        np.random.seed(self.manual_seed)

        # Initialize Dataset and DataLoader
        self.dataset = dataset
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=(split == 'train'),
            num_workers=opt.nThreads,
            pin_memory=True
        )

    def get_cpu_type(self, tensor_type):
        if tensor_type == 'torch.CudaHalfTensor':
            return torch.half
        elif tensor_type == 'torch.CudaDoubleTensor':
            return torch.double
        else:
            return torch.float32

    def size(self):
        return len(self.dataloader)

    def run(self):
        """
        Generator to yield batches from the DataLoader.
        """
        for batch in self.dataloader:
            inputs, targets = batch
            inputs = inputs.type(self.cpu_type)
            yield {'input': inputs, 'target': targets}


def create(opt):
    """
    Create train and validation data loaders.
    """
    loaders = {}
    
    # Define the transformations for the dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    for split in ['train', 'val']:
        dataset = CIFAR10(root='./data', train=(split == 'train'), download=True, transform=transform)
        loaders[split] = DataLoaderWrapper(dataset, opt, split)
    
    return loaders['train'], loaders['val']
