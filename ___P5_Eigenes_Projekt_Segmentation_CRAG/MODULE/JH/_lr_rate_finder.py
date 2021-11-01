# phyton3
# Letzte Änderung: 2021-09-08
# J.H

''' Content: Class
-------------------------------------------------------------------------------
LearningRateFinder  - 1. Train a model using different learning rates within
                        a range to find the optimal learning rate.
(Source: Learning rate finder according to Johannes Schmidt:
https://github.com/johschmidt42
Changes and adjustments to the processing of the data set CRAG_v2).                       
------------------------------------------------------------------------------
'''


import pandas as pd
import torch
from torch import nn
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm, trange
import math


class LearningRateFinder:
    '''
    1. Train a model using different learning rates within a range to find the optimal learning rate.
    '''
    
    def __init__(self,
                 model: nn.Module,
                 criterion,
                 optimizer,
                 device
                 ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.loss_history = {}
        self._model_init = model.state_dict()
        self._opt_init = optimizer.state_dict()
        self.device = device

    def fit(self,
            data_loader: torch.utils.data.DataLoader,
            steps=100,
            min_lr=1e-7,
            max_lr=1,
            constant_increment=False
            ):
        '''
        1.1 Trains the model for number of steps using varied learning rate and store the statistics
        '''
        self.loss_history = {}
        self.model.train()
        current_lr = min_lr
        steps_counter = 0
        epochs = math.ceil(steps / len(data_loader))
        progressbar = trange(epochs, desc='Progress')
        for epoch in progressbar:
            batch_iter = tqdm(enumerate(data_loader), 'Training', total=len(data_loader),
                              leave=False)
            for i, (x, y) in batch_iter:
                x, y = x.to(self.device), y.to(self.device)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr
                self.optimizer.zero_grad()
                out = self.model(x)
                loss = self.criterion(out, y)
                loss.backward()
                self.optimizer.step()
                self.loss_history[current_lr] = loss.item()
                steps_counter += 1
                if steps_counter > steps:
                    break
                if constant_increment:
                    current_lr += (max_lr - min_lr) / steps
                else:
                    current_lr = current_lr * (max_lr / min_lr) ** (1 / steps)

    def plot(self,
             smoothing=True,
             clipping=True,
             smoothing_factor=0.1,
             y_lim=[1,10],
             fontsize=20,
             path=''
             ):
        '''
        1.2 Shows loss vs learning rate(log scale) in a matplotlib plot
        '''
        loss_data_0 = pd.Series(list(self.loss_history.values()))
        lr_list_0 = list(self.loss_history.keys())
        loss_data=loss_data_0
        lr_list=lr_list_0
        if smoothing:
            loss_data = loss_data.ewm(alpha=smoothing_factor).mean()
            loss_data = loss_data.divide(pd.Series(
                [1 - (1.0 - smoothing_factor) ** i for i in range(1, loss_data.shape[0] + 1)]))  # bias correction
        if clipping:
            loss_data = loss_data[10:-5]
            lr_list = lr_list[10:-5]
        # Diagramm zeigen und abspeichern
        plt.plot(lr_list, loss_data)
        titlesize=fontsize+2
        tickssize=fontsize-2
        plt.xscale('log')
        plt.xticks(fontsize=tickssize)
        plt.yticks(fontsize=tickssize)
        plt.ylim(y_lim) 
        titlesize=fontsize+2
        plt.title('Verlust=f(Lernrate)', fontsize=titlesize )
        plt.xlabel('Lernrate, Learning rate', fontsize=fontsize)
        plt.ylabel('Verlust, Loss (gleitender Mittelwert)', fontsize=fontsize)
        plt.grid(linestyle = '--', linewidth = 0.5)
        if path != '':
            plt.savefig(path, bbox_inches="tight")
        return loss_data_0, lr_list_0

    def reset(self):
        '''
        1.3 Resets the model and optimizer to its initial state
        '''
        self.model.load_state_dict(self._model_init)
        self.optimizer.load_state_dict(self._opt_init)
        print('Model and optimizer in initial state.')