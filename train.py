import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math

class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, opt):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.opt = opt

    def train(self, epoch, dataloader):
        self.model.train()
        top1_sum, top5_sum, loss_sum = 0.0, 0.0, 0.0
        N = 0

        print(f'=> Training epoch # {epoch}')
        for n, sample in enumerate(dataloader):
            inputs, targets = sample['input'].to(self.opt['device']), sample['target'].to(self.opt['device'])
            
            if self.opt['lrShape'] == 'cosine':
                lr = self.learning_rate_cosine(epoch, n, len(dataloader))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            top1, top5 = self.compute_score(outputs, targets)
            batch_size = inputs.size(0)
            top1_sum += top1 * batch_size
            top5_sum += top5 * batch_size
            loss_sum += loss.item() * batch_size
            N += batch_size

            print(f' | Epoch: [{epoch}][{n + 1}/{len(dataloader)}]  Loss: {loss.item():.4f}  Top1: {top1:.2f}  Top5: {top5:.2f}')
        
        return top1_sum / N, top5_sum / N, loss_sum / N

    def test(self, epoch, dataloader):
        self.model.eval()
        top1_sum, top5_sum, loss_sum = 0.0, 0.0, 0.0
        N = 0
        
        with torch.no_grad():
            for n, sample in enumerate(dataloader):
                inputs, targets = sample['input'].to(self.opt['device']), sample['target'].to(self.opt['device'])
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                top1, top5 = self.compute_score(outputs, targets)
                batch_size = inputs.size(0)
                top1_sum += top1 * batch_size
                top5_sum += top5 * batch_size
                loss_sum += loss.item() * batch_size
                N += batch_size
        
        print(f' * Finished epoch # {epoch}  Top1: {top1_sum / N:.2f}  Top5: {top5_sum / N:.2f}')
        return top1_sum / N, top5_sum / N

    def compute_score(self, outputs, targets):
        _, predictions = outputs.topk(5, 1, True, True)
        correct = predictions.eq(targets.view(-1, 1).expand_as(predictions))
        
        top1 = 1.0 - correct[:, :1].sum().item() / targets.size(0)
        top5 = 1.0 - correct[:, :5].sum().item() / targets.size(0)
        
        return top1 * 100, top5 * 100

    def learning_rate_cosine(self, epoch, iteration, total_batches):
        total_steps = self.opt['epochs'] * total_batches
        current_step = (epoch - 1) * total_batches + iteration
        lr = 0.5 * self.opt['lr'] * (1 + math.cos(math.pi * current_step / total_steps))
        return lr

# Example usage
# model = YourModel()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
# trainer = Trainer(model, criterion, optimizer, scheduler, {'device': 'cuda', 'lrShape': 'cosine', 'lr': 0.1, 'epochs': 100})
# train_loader = DataLoader(...)
# trainer.train(1, train_loader)
