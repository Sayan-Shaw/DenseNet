# Copyright (c) 2016, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import argparse
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch DenseNet Training Script')
    
    # ------------- General options --------------------
    parser.add_argument('--data', default='', type=str, help='Path to dataset')
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['imagenet', 'cifar10', 'cifar100'],
                        help='Dataset options: imagenet | cifar10 | cifar100')
    parser.add_argument('--manualSeed', default=0, type=int, help='Manually set RNG seed')
    parser.add_argument('--nGPU', default=1, type=int, help='Number of GPUs to use by default')
    parser.add_argument('--backend', default='cudnn', type=str, choices=['cudnn', 'cunn'], help='Backend type')
    parser.add_argument('--cudnn', default='fastest', type=str, choices=['fastest', 'default', 'deterministic'], 
                        help='CUDNN options')
    parser.add_argument('--gen', default='gen', type=str, help='Path to save generated files')
    parser.add_argument('--precision', default='single', type=str, choices=['single', 'double', 'half'], 
                        help='Precision: single | double | half')
    
    # ------------- Data options ------------------------
    parser.add_argument('--nThreads', default=2, type=int, help='Number of data loading threads')
    
    # ------------- Training options --------------------
    parser.add_argument('--nEpochs', default=0, type=int, help='Number of total epochs to run')
    parser.add_argument('--epochNumber', default=1, type=int, help='Manual epoch number (useful on restarts)')
    parser.add_argument('--batchSize', default=32, type=int, help='Mini-batch size (1 = pure stochastic)')
    parser.add_argument('--testOnly', default=False, action='store_true', help='Run on validation set only')
    parser.add_argument('--tenCrop', default=False, action='store_true', help='Ten-crop testing')
    
    # ------------- Checkpointing options ---------------
    parser.add_argument('--save', default='checkpoints', type=str, help='Directory to save checkpoints')
    parser.add_argument('--resume', default='none', type=str, help='Resume from the latest checkpoint')
    
    # ---------- Optimization options -------------------
    parser.add_argument('--LR', default=0.1, type=float, help='Initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--weightDecay', default=1e-4, type=float, help='Weight decay')
    parser.add_argument('--lrShape', default='multistep', type=str, choices=['multistep', 'cosine'],
                        help='Learning rate strategy')
    
    # ---------- Model options --------------------------
    parser.add_argument('--netType', default='resnet', type=str, choices=['resnet', 'preresnet'],
                        help='Network architecture')
    parser.add_argument('--depth', default=20, type=int, help='ResNet depth')
    parser.add_argument('--shortcutType', default='', type=str, choices=['A', 'B', 'C'], help='Shortcut type')
    parser.add_argument('--retrain', default='none', type=str, help='Path to model to retrain with')
    parser.add_argument('--optimState', default='none', type=str, help='Path to reload optimState from')
    parser.add_argument('--shareGradInput', default=False, action='store_true',
                        help='Share gradInput tensors to reduce memory usage')
    parser.add_argument('--optnet', default=False, action='store_true', help='Use optnet to reduce memory usage')
    parser.add_argument('--resetClassifier', default=False, action='store_true',
                        help='Reset the fully connected layer for fine-tuning')
    parser.add_argument('--nClasses', default=0, type=int, help='Number of classes in the dataset')
    
    # ---------- Model options for DenseNet -------------
    parser.add_argument('--growthRate', default=12, type=int, help='Number of output channels at each conv layer')
    parser.add_argument('--bottleneck', default=True, action='store_true',
                        help='Use 1x1 convolution to reduce dimension (DenseNet-B)')
    parser.add_argument('--reduction', default=0.5, type=float, help='Channel compression ratio')
    parser.add_argument('--dropRate', default=0, type=float, help='Dropout probability')
    parser.add_argument('--optMemory', default=2, type=int, help='Optimize memory for DenseNet: 0-5')
    parser.add_argument('--d1', default=0, type=int, help='Number of layers in block 1')
    parser.add_argument('--d2', default=0, type=int, help='Number of layers in block 2')
    parser.add_argument('--d3', default=0, type=int, help='Number of layers in block 3')
    parser.add_argument('--d4', default=0, type=int, help='Number of layers in block 4')
    
    opt = parser.parse_args()
    
    # Post-processing
    if opt.dataset == 'imagenet':
        if not os.path.isdir(opt.data):
            sys.exit('Error: missing ImageNet data directory')
        train_dir = os.path.join(opt.data, 'train')
        if not os.path.isdir(train_dir):
            sys.exit('Error: ImageNet missing `train` directory')
        opt.shortcutType = opt.shortcutType or 'B'
        opt.nEpochs = opt.nEpochs or 90
    
    elif opt.dataset in ['cifar10', 'cifar100']:
        opt.shortcutType = opt.shortcutType or 'A'
        opt.nEpochs = opt.nEpochs or 164
    
    else:
        sys.exit(f'Unknown dataset: {opt.dataset}')
    
    if opt.precision == 'single':
        opt.tensorType = torch.float32
    elif opt.precision == 'double':
        opt.tensorType = torch.float64
    elif opt.precision == 'half':
        opt.tensorType = torch.float16
    else:
        sys.exit(f'Unknown precision: {opt.precision}')
    
    if opt.resetClassifier and opt.nClasses == 0:
        sys.exit('-nClasses required when resetClassifier is set')
    
    if opt.shareGradInput and opt.optnet:
        sys.exit('Error: cannot use both --shareGradInput and --optnet')
    
    return opt


if __name__ == '__main__':
    options = parse_args()
    print(options)
