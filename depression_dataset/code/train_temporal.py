import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle as pkl
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
import math
import random
from pathlib import Path
from tqdm import tqdm
plt.style.use('seaborn')


class LSTM(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(LSTM, self).__init__()
        self.hid_dim = hid_dim

        self.lstm = nn.LSTM(in_dim, hid_dim)
        self.linear = nn.Linear(hid_dim, out_dim)
        self.hidden_cell = (torch.zeros(1,1,self.hid_dim),
                            torch.zeros(1,1,self.hid_dim))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1)).double()
        return predictions[-1]

def split_folds(batches, n_fold):
    folds = []
    fold_size = math.trunc((len(batches)/n_fold))
    for i in range(0, n_fold):
        start = i*fold_size
        end = (i+1)*fold_size
        fold = batches[start:end]
        folds.append(fold)
    return folds

def compute_valid_losses(model, loss_func, data):
  losses = []
  for seq, labels in data:
    model.hidden_cell = (torch.zeros(1, 1, model.hid_dim),
                    torch.zeros(1, 1, model.hid_dim))
    y_pred = model(seq)
    loss = torch.sqrt(loss_func(y_pred, labels))
    losses.append(loss.item())
  return(losses)

def compute_test_score(model, loss_func, data):
    losses = []
    for seq, labels in data:
        model.hidden_cell = (torch.zeros(1, 1, model.hid_dim),
                     torch.zeros(1, 1, model.hid_dim))
        y_pred = model(seq)
        loss = abs((y_pred-labels).item()) # MAYBE MATH ABS?
        losses.append(loss)
    mean = np.mean(losses)
    std = np.std(losses)
    return(mean, std, losses)


def create_batches(x,y):
    y = torch.tensor(y.values).double()
    batches = []
    for i, xi in enumerate(x):
        xi = xi.astype(float)
        xi = torch.tensor(xi.values)
        xi = xi.view(xi.shape[0], 1, xi.shape[1]).float()
        yi = y[i]
        batches.append((xi, yi))
    random.shuffle(batches)
    return(batches)



def cross_validate(x, y, n_fold, no_epochs):
    batches = create_batches(x,y)
    train_size  = int(8 * len(batches) / 10)
    train_batches = batches[:train_size]
    test_batches = batches[train_size:]

    batches = train_batches
    no_features = batches[0][0].shape[2]
    folds = split_folds(batches, n_fold)
    rmse = []
    loss_func = nn.MSELoss()

    for f in range(len(folds)):
        print('--- training fold {} ---'.format(f))

        model = LSTM(in_dim=no_features, hid_dim=50, out_dim=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Prepare fold data
        valid = folds[f]
        train = []
        for j in range(len(folds)):
            if j != f:
                train.append(folds[j])
        train = list(itertools.chain(*train))

        # Fold losses for early stopping
        fold_train_losses = []
        fold_valid_losses = []

        # Optimize fold
        for epoch in tqdm(range(no_epochs)):
            #print(fold_valid_losses)
            window_size = 3 #8
            treshold = 0.005#0.005

            if epoch > window_size*2:
                old_window = np.mean(fold_valid_losses[-window_size*2:-window_size])
                new_window = np.mean(fold_valid_losses[-window_size:])

                if new_window > old_window:
                    break

            epoch_train_losses = []
            epoch_valid_losses = []
            for seq, labels in train:
                optimizer.zero_grad()
                model.hidden_cell = (torch.zeros(1, 1, model.hid_dim),
                                torch.zeros(1, 1, model.hid_dim))
                y_pred = model(seq)
                loss = torch.sqrt(loss_func(y_pred, labels))
                loss.backward()
                optimizer.step()
                epoch_train_losses.append(loss.item())

            epoch_valid_losses = compute_valid_losses(model, loss_func, valid)
            epoch_train_loss = round(np.mean(epoch_train_losses), 4)
            epoch_valid_loss = round(np.mean(epoch_valid_losses), 4)
            fold_train_losses.append(epoch_train_loss)
            fold_valid_losses.append(epoch_valid_loss)

        rmse.append(epoch_valid_loss)

        # Plot fold convergence
        plt.plot(fold_train_losses)
        plt.plot(fold_valid_losses)
        plt.xlabel('epoch')
        plt.ylabel('RMSE loss')
        plt.legend(['train fold','validation fold'], loc='upper right')
        Path("results/temporal_model").mkdir(parents=True, exist_ok=True)
        plt.savefig('results/temporal_model/fold_loss_fold_{}.png'.format(f))
        plt.clf()

    print('validation loss mean: {}'.format(np.mean(rmse)))

    mean, std, all_distances = compute_test_score(model, loss_func, test_batches)
    print('test loss mean: {}'.format(mean))
    print('test loss std: {}'.format(std))
    print('absolute distances: {}'.format(all_distances))
