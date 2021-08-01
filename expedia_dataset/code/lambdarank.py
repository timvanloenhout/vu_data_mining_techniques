import torch
import torch.nn as nn
import pickle as pkl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm

import argparse
from sklearn.metrics import ndcg_score
from scipy import stats
from sklearn.utils import resample
import lambdarank_model
from tensorflow import keras
from keras import backend as K
from keras.layers import Activation, Dense, Input, Subtract
from keras.models import Model

def create_ranking(scored_list):
    print('Creating ranking...')
    submission = scored_list.sort_values(['srch_id','score'], ascending=[True, False])[['srch_id', 'prop_id']]
    submission = submission.astype(int)
    print(submission)
    submission.to_csv('submission_xgb.csv', index=False)

def predict_test_set(model, x, qid, prop_id):
    print('Predicting test set...')
    y_pred = model.predict(x).reshape((-1,1))
    qid = qid.reshape((-1,1))
    prop_id = prop_id.reshape((-1,1))
    test_results = np.hstack((qid, prop_id, y_pred))
    test_results = pd.DataFrame(data=test_results, columns=['srch_id', 'prop_id', 'score'])
    create_ranking(test_results)



def remove_features(data, rem_features):
    for f in rem_features:
        data = data[data.columns.drop(list(data.filter(regex=f)))]
    return(data)

def upsample(df):
    # Separate majority and minority classes
    df_0 = df[df['012']==0]
    df_1 = df[df['012']==1]
    df_2 = df[df['012']==2]

    # Upsample minority classes
    df_1 = resample(df_1, replace=True, n_samples=len(df_0)*int(str(ARGS.upsample_ratio[1])), random_state=42)
    df_2 = resample(df_2, replace=True, n_samples=len(df_0)*int(str(ARGS.upsample_ratio[2])), random_state=42)

    # Concatenate majority class with upsampled minority classes
    df_upsampled = pd.concat([df_0, df_1, df_2])
    col_names = df_upsampled.columns
    df_upsampled = df_upsampled.to_numpy()
    df_upsampled = df_upsampled[df_upsampled[:,0].argsort()]
    df_upsampled = pd.DataFrame(df_upsampled)
    df_upsampled.columns = col_names

    print('after upsampling')
    print(df_upsampled['012'].value_counts())

    return(df_upsampled)


def train_model(train_x, train_y, train_qid, val_x, val_y, val_qid):
    #model = lambdarank.LambdaRankNN(input_size=train_x.shape[1], hidden_layer_sizes=(64,32,16,), activation=('relu', 'relu', 'relu'), solver='adam')
    model = lambdarank.LambdaRankNN(input_size=train_x.shape[1], hidden_layer_sizes=(16,8,), activation=('relu', 'relu'), solver='adam')
    n_splits = int(train_x.shape[0]/ARGS.batch_size)
    best_ndcg = 0
    iter = 0
    while iter < ARGS.max_n_iters:
        print('iter', iter)
        for s in range(0,n_splits):
            # NOW CUT-OFF CAN BE IN MIDDLE OF SRCH_ID, SO NEED TO FIND A SMARTER WAY
            start = ARGS.batch_size*s
            end = ARGS.batch_size*(s+1)

            train_y_sub = train_y[start:end]
            train_qid_sub = train_qid[start:end]
            train_x_sub = train_x[start:end]

            model.fit(train_x_sub, train_y_sub, train_qid_sub, epochs=1)
            model.evaluate(train_x, train_y, train_qid, eval_at=5)
            valid_ndcg = model.evaluate(val_x, val_y, val_qid, eval_at=5, valid=True)

            iter+=1
            if iter == ARGS.max_n_iters:
                break

    return(model)

def normalize(x):
    x = (x-np.min(x, axis=0))/(np.max(x, axis=0)-np.min(x, axis=0))
    return(x)

def normalize_price_per_country(x):
    for feature in ['price_per_person', 'price_per_adult', 'price_usd']:
        countries = {}
        for region, country in x.groupby('prop_country_id'):
            name = country['prop_country_id'].iloc[0]
            max_value = country[feature].max()
            min_value = country[feature].min()
            maxmin_value = max_value-min_value
            info = {}
            if maxmin_value > 0:
                info['maxmin'] = maxmin_value
            else:
                info['maxmin'] = 1
            info['min'] = min_value
            countries[name] = info
        x[feature] = x.apply(lambda row: (row[feature] - countries[row['prop_country_id']]['min']) /
                                    countries[row['prop_country_id']]['maxmin'], axis=1)
        print(x[feature].isnull().values.any())

    return(x)

def normalize_price_per_query(x):
    for feature in ['price_per_person', 'price_per_adult', 'price_usd']:
        print(feature)
        countries = {}
        for region, country in x.groupby('srch_id'):
            name = country['srch_id'].iloc[0]
            max_value = country[feature].max()
            min_value = country[feature].min()
            maxmin_value = max_value-min_value
            info = {}
            if maxmin_value > 0:
                info['maxmin'] = maxmin_value
            else:
                info['maxmin'] = 1
            info['min'] = min_value
            countries[name] = info
        x[feature] = x.apply(lambda row: (row[feature] - countries[row['srch_id']]['min']) /
                                    countries[row['srch_id']]['maxmin'], axis=1)

    return(x)

def prepare_data(x, y, test_x):
    # Upsample clicks and bookings
    if ARGS.upsample == True:
        y = y[y.columns.drop(list(y.filter(regex='srch_id')))]
        xy = pd.concat([x, y], axis=1)
        xy = upsample(xy)
        x = xy.iloc[:,:x.shape[1]]
        y = xy.iloc[:,x.shape[1]:]

    # Create y
    y.loc[y['012'] == 0, '015'] = int(0)
    y.loc[y['012'] == 1, '015'] = int(1)
    y.loc[y['012'] == 2, '015'] = int(5)
    y = y[[ARGS.label]].to_numpy().flatten()

    # Create qid
    qid = x[['srch_id']].to_numpy().flatten()

    # Create x
    used_features = [c for c in list(x) if len(x[c].unique()) > 1]
    x = x[used_features]
    #x = normalize_price_per_query(x)
    #x = normalize_price_per_country(x)
    x = remove_features(x, ARGS.remove_features)
    col_names = x.columns
    x = x.to_numpy()

    if ARGS.normalize == True:
        x = normalize(x)

    # Split into training and validation set
    train_size = int(x.shape[0]*0.8)
    train_x = x[:train_size]
    train_y = y[:train_size]
    train_qid = qid[:train_size]
    val_x = x[train_size:]
    val_y = y[train_size:]
    val_qid = qid[train_size:]






    # Remove missing value rows from train_data
    train_x = pd.DataFrame(columns=col_names, data=train_x)
    used_rows = train_x['random_bool']==0
    print(train_x.shape)
    train_x = train_x[used_rows]
    train_qid = train_qid[used_rows]
    train_y = train_y[used_rows]
    print(train_x.shape)

    train_x = train_x.to_numpy()






    #Remove outliers
    if ARGS.outlier_reduction == True:
        non_outliers = (np.abs(stats.zscore(train_x)) < ARGS.z).all(axis=1)
        train_x = train_x[non_outliers]
        train_y = train_y[non_outliers]
        train_qid = train_qid[non_outliers]

    # Create test data x
    test_qid = 0 # for return in case of no test
    test_prop_id = 0
    if ARGS.test == True:
        test_qid = test_x[['srch_id']].copy().to_numpy().flatten()
        test_prop_id = test_x[['prop_id']].copy().to_numpy().flatten()
        test_x = test_x[used_features]
        #test_x = normalize_price_per_query(test_x)
        #test_x = normalize_price_per_country(test_x)
        test_x = remove_features(test_x, ARGS.remove_features)
        test_x = test_x.to_numpy()
        if ARGS.normalize == True:
                test_x = normalize(test_x)

    return(train_x, train_y, train_qid, val_x, val_y, val_qid, test_x, test_qid, test_prop_id)


def main():
    # Load data
    y = pd.read_pickle('../data/newest_y.pkl')[:1000000]
    x = pd.read_pickle('../data/newest_x.pkl')[:1000000]
    test_x = pd.read_pickle('../data/newest_test.pkl')#[0:1000]


    test_x['part_of_day_0']=test_x['parf_of_day_0']
    test_x['part_of_day_1']=test_x['parf_of_day_1']
    test_x['part_of_day_2']=test_x['parf_of_day_2']
    test_x['part_of_day_3']=test_x['parf_of_day_3']

    filter_comp = [col for col in test_x if col.startswith('parf_of_day')]
    test_x = test_x.drop(filter_comp, axis=1)

    # Remove seasons
    filter_comp = [col for col in x if col.startswith('season')]
    x = x.drop(filter_comp, axis=1)
    test_x = test_x.drop(filter_comp, axis=1)

    # Prepare data
    train_x, train_y, train_qid, val_x, val_y, val_qid, test_x, test_qid, test_prop_id = prepare_data(x, y, test_x)

    # Train model
    model = train_model(train_x, train_y, train_qid, val_x, val_y, val_qid)


    # Run model on test set
    if ARGS.test == True:
        predict_test_set(model, test_x, test_qid, test_prop_id)





if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--label', default='dcg', type=str,
                        help='prediction target')
    PARSER.add_argument('--normalize', default=False, type=bool,
                        help='normalize features')
    PARSER.add_argument('--remove_features', default=['visitor_loc_country', 'prop_id', 'srch_id', 'prop_country_id'], type=list,
                        help='features to be removed before training')
    PARSER.add_argument('--max_n_iters', default=20, type=int,
                        help='number of iterations')
    PARSER.add_argument('--batch_size', default=200000, type=int,
                        help='number of entries per batch')
    PARSER.add_argument('--test', default=False, type=bool,
                        help='predict test set and create ranking')
    PARSER.add_argument('--outlier_reduction', default=False, type=bool,
                        help='Remove outlier entries')
    PARSER.add_argument('--z', default=3, type=int,
                        help='z-value threshold for outlier reduction')
    PARSER.add_argument('--upsample', default=False, type=bool,
                        help='usample clicks and bookings')
    PARSER.add_argument('--upsample_ratio', default='111', type=str,
                        help='upsample clicks and bookings')


    ARGS = PARSER.parse_args()

    main()
