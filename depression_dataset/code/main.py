import data_utils as data_utils
#import results_utils as results_utils


from sklearn.feature_selection import f_regression

import torch
import argparse
import sys
import pickle as pkl
from pathlib import Path

import train_temporal as train_temporal
from train_standard import *

# -----------------------------------------------------------
# Select device and show information
# -----------------------------------------------------------

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('----------------------------------')
print('Using device for training:', DEVICE)
print('----------------------------------')
print()


def temporal():
    if ARGS.load:
        with open(f"./data/x_temporal_{ARGS.suffix}.pkl", "rb") as f:
            x_temporal = pkl.load(f)
        with open(f"./data/x_basic_{ARGS.suffix}.pkl", "rb") as f:
            x_basic = pkl.load(f)
        with open(f"./data/y_{ARGS.suffix}.pkl", "rb") as f:
            y = pkl.load(f)
    else:
        x_basic, x_temporal, y = data_utils.data_preprocessing(ARGS)


    # TEMPORAL MODEL
    # train_temporal.cross_validate(x_temporal, y, ARGS.no_folds,
    #                                    ARGS.max_no_epochs_temporal_model)


    data_utils.univariate_selection(x_basic, y, f_regression)
    #data_utils.correlation_heatmap(x_basic, y)





def test_param():
    if ARGS.load:

        with open(f"./data/x_basic_{ARGS.suffix}.pkl", "rb") as f:
            x_basic = pkl.load(f)

        with open(f"./data/y_{ARGS.suffix}.pkl", "rb") as f:
            y = pkl.load(f)
    else:
        x_basic, _, y = data_utils.data_preprocessing(ARGS)

    train_x = x_basic[:1140].fillna(0).to_numpy() # 8:2 split would be 1014:253
    train_y = y[:1140].fillna(0).to_numpy()


    # create/load datasets
    #x_basic, x_temporal, y = data_utils.data_preprocessing(ARGS)

    with open(f"data/x_basic_{ARGS.suffix}.pkl", "rb") as f:
        x_basic = pkl.load(f)
    with open(f"data/x_temporal_{ARGS.suffix}.pkl", "rb") as f:
        x_temporal = pkl.load(f)
    with open(f"data/y_{ARGS.suffix}.pkl", "rb") as f:
        y = pkl.load(f)

    param = [1e-3, 1e-2, 1e-1, 0.12, 0.14, 0.16, 2e-1 ]
    t, v, = test_param(train_x, train_y, param)
    print(len(t))
    plot_rmse(t, v)

    # # TEMPORAL MODEL
    train_temporal.cross_validate(x_temporal, y, ARGS.no_folds,
                                       ARGS.max_no_epochs_temporal_model)

def benchmark():
    x, y = data_utils.benchmark_data()

    train_x = x[:1140].to_numpy()
    train_y = y[:1140].to_numpy()

    test_x = x[1140:].to_numpy()
    test_y = y[1140:].to_numpy()

    pred = benchmark_predict(train_x)
    mse, rmse = eval_mse(pred, train_y)
    r2, r2_adjusted = eval_r2(pred, train_y)
    print(mse, rmse)
    print(r2, r2_adjusted)


def standard():
    """
    So far only containing the standard model timeline, later divide into
    standard/temporal
    """
    # create/load datasets
    if ARGS.load:

        with open(f"./data/x_basic_{ARGS.suffix}.pkl", "rb") as f:
            x_basic = pkl.load(f)

        with open(f"./data/y_{ARGS.suffix}.pkl", "rb") as f:
            y = pkl.load(f)
    else:
        x_basic, _, y = data_utils.data_preprocessing(ARGS)

    train_x = x_basic[:1140].fillna(0).to_numpy() # 8:2 split would be 1014:253
    train_y = y[:1140].fillna(0).to_numpy()

    # test_x = x_basic[1140:]
    # test_y = y[1140:]

    _, avg_rmse = svm_train(train_x, train_y)

# -----------------------------------------------------------
# Compile ARGS and run main()
# -----------------------------------------------------------
if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--model', default='temporal', type=str,
                        help='either the standard or temporal')
    PARSER.add_argument('--no_folds', default=10, type=int,
                        help='number of cross validation folds')
    PARSER.add_argument('--max_no_epochs_temporal_model', default=50, type=int,
                        help='maximum number of epochs for the temporal model')

    PARSER.add_argument('--suffix', default='basic', type=str,
                        help="short name that will be added to filenames")

    PARSER.add_argument('--test_param', default=False, type=bool,
                        help="to test svm parameters")
    PARSER.add_argument("--load", default=True, type=bool,
                        help="whether to load processed from file or not")





    PARSER.add_argument('--window_size', default=5, type=int, help='number of days in the time window')

    # EXTRA ARG FOR TESTING. IF SET TO VERY HIGH, SAY 100, YOU TEST WITHOUT OUTLIER REDUCTION
    PARSER.add_argument('--z_thresh', default=2, type=int, help='no of standard deviations for outlier reduction')

    PARSER.add_argument('--wa_pod_missing', default='sum', type=str,
                        help='window aggregation metric for part of day and missing value feature')

    PARSER.add_argument('--wa_std', default=False, type=bool,
                        help='add standard deviation to features with mean as window aggregation metric')

    PARSER.add_argument('--wa_weighted', default=False, type=bool,
                        help='use weighted window aggregation')

    PARSER.add_argument('--use_pod_features', default=False, type=bool,
                        help='use part of day features')

    PARSER.add_argument('--use_missing_features', default=False, type=bool,
                        help='use missing value features')

    PARSER.add_argument('--remove_standard_features', default=[], type=list,
                                        help='features to be removed before training')

    PARSER.add_argument('--normalize', default=True, type=str,
                        help='normalize data')


    ARGS = PARSER.parse_args()

    if ARGS.test_param:
        test_param()
    else:
        if ARGS.model == "standard":
            standard()
        elif ARGS.model == "temporal":
            temporal()
        elif ARGS.model == "benchmark":
            benchmark()
