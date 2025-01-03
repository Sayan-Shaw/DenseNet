# Copyright (c) 2016, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import torch
import random
import numpy as np
from dataloader import create_dataloader
from models import setup_model
from train import Trainer
from opts import parse_opts
from checkpoints import latest_checkpoint, save_checkpoint

# Set default tensor type and seeds
torch.set_default_dtype(torch.float32)
torch.set_num_threads(1)

# Parse command-line options
opt = parse_opts()
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)

# Load previous checkpoint, if it exists
checkpoint, optim_state = latest_checkpoint(opt)

# Create model and criterion
model, criterion = setup_model(opt, checkpoint)

# Data loading
train_loader, val_loader = create_dataloader(opt)

# The trainer handles the training loop and evaluation on validation set
trainer = Trainer(model, criterion, opt, optim_state)

if opt.testOnly:
    top1_err, top5_err = trainer.test(0, val_loader)
    print(f' * Results top1: {top1_err:.3f}  top5: {top5_err:.3f}')
    exit()

# Start training
start_epoch = checkpoint['epoch'] + 1 if checkpoint else opt.epochNumber
best_top1 = float('inf')
best_top5 = float('inf')

for epoch in range(start_epoch, opt.nEpochs + 1):
    # Train for a single epoch
    train_top1, train_top5, train_loss = trainer.train(epoch, train_loader)
    
    # Run model on validation set
    test_top1, test_top5 = trainer.test(epoch, val_loader)

    best_model = False
    if test_top1 < best_top1:
        best_model = True
        best_top1 = test_top1
        best_top5 = test_top5
        print(' * Best model ', test_top1, test_top5)

    save_checkpoint(epoch, model, trainer.optim_state, best_model, opt)

print(f' * Finished top1: {best_top1:.3f}  top5: {best_top5:.3f}')
