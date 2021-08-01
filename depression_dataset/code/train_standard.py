import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import pickle as pkl
from tqdm import tqdm

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

#from standard_model import MLP

########################
### HELPER FUNCTIONS ###
########################
def round_robin_split(features, labels, n_fold):
    split = list()
    for i in range(n_fold):
        split.append((features[i:len(features):n_fold], labels[i:len(features):n_fold]))

    return split


def avg(l):
    return sum(l)/len(l)


def eval_mse(prediction, labels):
    mse = mean_squared_error(labels, prediction)
    rmse = mean_squared_error(labels, prediction, squared=False)
    return mse, rmse

def eval_r2(prediction, labels):
    r2 = r2_score(labels, prediction)
    n = len(labels)
    k = 41
    r2_adj = 1 - (1 - r2)*(n-1)/(n-41-1)
    return r2, r2_adj

#################
### SVR MODEL ###
#################
def svm_train(features, labels, suffix, n_fold=6):
    folds = round_robin_split(features, labels, n_fold)

    train_rmse_list = list()
    valid_rmse_list = list()
    models = list()

    for i in tqdm(range(len(folds))):
        new_fold = True
        valid_X, valid_y = folds[i]
        for j in range(len(folds)):
            if j != i:
                if new_fold:
                    train_X, train_y = folds[j]
                    new_fold = False
                else:
                    train_X = np.append(train_X, folds[j][0], axis=0)
                    train_y = np.append(train_y, folds[j][1], axis=0)

        clf = SVR()
        clf.fit(train_X, train_y.squeeze())

        pred = clf.predict(train_X)
        train_rmse = mean_squared_error(train_y.squeeze(), pred, squared=False) # returns RMSE
        # mse_list = (np.square(np.subtract(y.squeeze(),pred))) # returns SE

        pred = clf.predict(valid_X)
        valid_rmse = mean_squared_error(valid_y.squeeze(), pred, squared=False) # returns RMSE

        train_rmse_list.append(train_rmse)
        valid_rmse_list.append(valid_rmse)

        models.append(clf)

    valid_avg_rmse = avg(valid_rmse_list)
    train_avg_rmse = avg(train_rmse_list)


    best_model = models[valid_rmse.index(min(valid_rmse))]

    filename = f"./data/basic_model/svm_{suffix}.mdl"
    with open(filename, "wb") as f:
        pkl.dump(best_model, f)

    return train_avg_rmse, valid_avg_rmse


def svm_predict(features, suffix):
    filename = f"./data/basic_model/svm_{suffix}.mdl"
    with open(filename, "rb") as f:
        clf = pkl.load(f)

    return clf.predict(features)


#######################
### BENCHMARK MODEL ###
#######################
def benchmark_train(features, labels, n_fold=6):
    folds = round_robin_split(features, labels, n_fold)

    mse = list()

    for i in tqdm(range(len(folds))):
        valid_x, valid_y = folds[i]

        fold_mse = mean_squared_error(valid_y.squeeze(), valid_x, squared=False) # returns RMSE

        mse.append(fold_mse)

    avg_mse = avg(mse)

    return avg_mse


def benchmark_predict(features):
    return features



#################################
### TESTING WITH SVR MAX_ITER ###
#################################
def test_param(features, labels, param_list):
    folds = round_robin_split(features, labels, 6)

    param_train = dict()
    param_valid = dict()

    for mit in param_list:
        param_train[mit] = []
        param_valid[mit] = []

    for i in range(len(folds)):
        new_fold = True
        valid_tup = folds[i]
        for j in range(len(folds)):
            if j != i:
                if new_fold:
                    train_X, train_y = folds[j]
                    new_fold = False
                else:
                    train_X = np.append(train_X, folds[j][0], axis=0)
                    train_y = np.append(train_y, folds[j][1], axis=0)


        for mit in tqdm(param_list):
            clf = SVR(tol=mit)
            clf = clf.fit(train_X, train_y.squeeze())
            train_pred = clf.predict(train_X)
            valid_pred = clf.predict(valid_tup[0])

            param_train[mit].append(mean_squared_error(train_y.squeeze(), train_pred, squared=False))
            param_valid[mit].append(mean_squared_error(valid_tup[1].squeeze(), valid_pred, squared=False))

    return param_train, param_valid


def plot_rmse(train, valid):

    t = []
    v = []
    x = list(train.keys())
    for i in train:
        t.append(avg(train[i]))

    for l in valid:
        v.append(avg(valid[l]))
    # e = []
    # for l in t:
    #     e.append(l* (1140+139)/(1140-139))

    plt.style.use('seaborn')
    plt.plot(x, t, label="train avg rmse over all folds")
    plt.plot(x, v, label="valid avg rmse over all folds")
    # plt.plot(x, e, label="expected test rmse")

    plt.xlabel("max_iter")
    plt.ylabel("RMSE")
    plt.title("Train and Validation average RMSE")
    plt.legend()
    plt.show()
    plt.savefig("svm convergence")


############################
### NEURAL NETWORK MODEL ###
############################
def get_batches(features, labels, batch_size):
    shuffle_idx = torch.randperm(len(features)) # shuffle the indices
    i = 0
    while i < len(features):
        idx = shuffle_idx[i:i+batch_size]
        yield features[idx], labels[idx]
        i += batch_size


def eval_nn(model, valid_tup, device):
    valid_feature = torch.from_numpy(valid_tup[0]).float().to(device)
    valid_label = valid_tup[1].squeeze()
    pred = model(valid_feature).detach().numpy().squeeze()
    return mean_squared_error(pred, valid_label, squared=False)


def train_nn(features, labels, valid_tup, batch_size, device, lr, eval_every, patience):

    model = MLP(213, 128)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    features = torch.from_numpy(features).float().to(device)
    labels = torch.from_numpy(labels).float().to(device)

    start_time = time.time()
    criterion = nn.MSELoss()

    step = 0
    wait = 0
    best_loss = np.inf

    train = True
    model.train()
    while train:
        for x, y in get_batches(features, labels, batch_size):
            out = model(x)
            loss = torch.sqrt(criterion(out, y)) # RMSE

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if step % eval_every == 0:
                model.eval()
                valid_loss = eval_nn(model, valid_tup, device)
                model.train()

                print(f"\tStep: {step}, loss: {valid_loss:.2f},",
					  f"time trained: {time.time() - start_time:.3f}.")

                if valid_loss < best_loss:
                    print(f"\tNew lowest loss")
                    best_loss = valid_loss
                    best_model = model.state_dict()
                    wait = 0
                else:
                    wait += 1
                    if wait > patience:
                        print("Stopping early")
                        train = False
                        break

            step += 1

    return best_loss, best_model
