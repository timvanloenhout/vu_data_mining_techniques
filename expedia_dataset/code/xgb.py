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

from sklearn.metrics import ndcg_score
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor, XGBRanker

def create_ranking(scored_list):
    print('creating ranking...')
    submission = scored_list.sort_values(['srch_id','score'], ascending=[True, False])[['srch_id', 'prop_id']]
    submission = submission.astype(int)
    print(submission)
    submission.to_csv('submission_xgb.csv', index=False)

def remove_features(data, rem_features):
    for f in rem_features:
        data = data[data.columns.drop(list(data.filter(regex=f)))]
    return(data)



def predict_test_set(model, x, qid, prop_id):
    print('predicting test set...')
    y_pred = model.predict(x).reshape((-1,1))
    qid = qid.reshape((-1,1))
    prop_id = prop_id.reshape((-1,1))
    test_results = np.hstack((qid, prop_id, y_pred))
    test_results = pd.DataFrame(data=test_results, columns=['srch_id', 'prop_id', 'score'])
    create_ranking(test_results)



def normalize(x):
    x = (x-np.min(x, axis=0))/(np.max(x, axis=0)-np.min(x, axis=0))
    return(x)

def normalize_price_per_query(x):
    print('normalizing per query...')
    col_names = list(x.columns)
    col_names.remove('srch_id')
    for feature in tqdm(col_names):
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
        # print(x[feature].isnull().values.any())
    return(x)


def keep_rand_rankings(train_x, train_y, train_qid, col_names):
    train_x = pd.DataFrame(columns=col_names, data=train_x)
    used_rows = train_x['random_bool']==0
    print('all rankings: ', train_x.shape)
    train_x = train_x[used_rows]
    train_qid = train_qid[used_rows]
    train_y = train_y[used_rows]
    print('random rankings: ', train_x.shape)

    train_x = train_x.to_numpy()
    return(train_x, train_y, train_qid)


def remove_missing_value_rows(train_x, train_y, train_qid, col_names):
    print('removing rows containing missing values...')
    train_x = pd.DataFrame(columns=col_names, data=train_x)
    missing_features = ['missing_visitor_hist_starrating', 'missing_visitor_hist_adr_usd',
       'missing_prop_starrating', 'missing_srch_query_affinity_score'] # 'missing_prop_review_score',

    for f in missing_features:
        used_rows = train_x[f]==0
        print('{}: {}'.format(f, train_x.shape))
        train_x = train_x[used_rows]
        train_qid = train_qid[used_rows]
        train_y = train_y[used_rows]
        print('{}: {}'.format(f, train_x.shape))

    train_x = train_x.to_numpy()

    return(train_x, train_y, train_qid)



def upsample(train_x, train_y, train_qid, col_names):
    train_x = pd.DataFrame(columns=col_names, data=train_x)
    train_y = pd.DataFrame(columns=['012', 'label'], data=train_y)
    xyq = pd.concat([train_qid.reset_index()['srch_id'], train_x, train_y], axis=1)

    # Separate majority and minority classes
    xyq_0 = xyq[xyq['012']==0]
    xyq_1 = xyq[xyq['012']==1]
    xyq_2 = xyq[xyq['012']==2]

    # Upsample minority classes
    xyq_1 = resample(xyq_1, replace=True, n_samples=len(xyq_0)*int(str(ARGS.upsample_ratio[1])), random_state=42)
    xyq_2 = resample(xyq_2, replace=True, n_samples=len(xyq_0)*int(str(ARGS.upsample_ratio[2])), random_state=42)

    # Concatenate majority class with upsampled minority classes
    xyq = pd.concat([xyq_0, xyq_1, xyq_2])
    col_names = xyq.columns
    xyq = xyq.to_numpy()
    xyq = xyq[xyq[:,0].argsort()]
    xyq = pd.DataFrame(xyq)
    xyq.columns = col_names

    print('ratio after upsampling')
    print(xyq['012'].value_counts())

    train_qid = xyq.iloc[:,0]
    train_x = xyq.iloc[:,1:train_x.shape[1]+1].to_numpy()
    train_y = xyq.iloc[:,train_x.shape[1]+1:].to_numpy()

    return(train_x, train_y, train_qid)

def remove_outliers(train_x, train_y, train_qid):
    print('removing outliers...')
    print('all rows ', train_x.shape[0])
    outliers = (np.abs(stats.zscore(train_x)) < ARGS.z)
    outliers = (outliers == False).sum(axis=1)
    n_outliers = np.array(np.unique(outliers, return_counts=True)).T
    outliers = outliers<=ARGS.n_outlier_features
    train_x = train_x[outliers]
    train_y = train_y[outliers]
    train_qid = train_qid[outliers]
    print('remaining rows ', train_x.shape[0])
    print(n_outliers)

    return(train_x, train_y, train_qid)



def prepare_data(x, y, test_x):
    # Location data
    x = remove_features(x, ['visitor_location_country_id', 'prop_country_id'])
    # x['visitor_location_country_id'] = x['visitor_location_country_id'].astype('str')
    # x['prop_country_id'] = x['prop_country_id'].astype('str')

    # New missing feature approach
    # x = x.drop(columns=['prop_review_score', 'prop_starrating', 'srch_query_affinity_score'])
    # prs = pd.read_pickle('../data/all_prop_review_score.pkl')[:500000]
    # sqa = pd.read_pickle('../data/all_srch_query_affinity_score_500000.pkl')[:500000]
    # ps = pd.read_pickle('../data/all_prop_starrating.pkl')[:500000]
    # new_missing = pd.concat([prs, sqa, ps], axis=1)
    # new_missing.columns = ['prs', 'sqa', 'ps']
    # x = pd.concat([x, new_missing], axis=1)

    # Fix part of day names
    test_x['part_of_day_0']=test_x['parf_of_day_0']
    test_x['part_of_day_1']=test_x['parf_of_day_1']
    test_x['part_of_day_2']=test_x['parf_of_day_2']
    test_x['part_of_day_3']=test_x['parf_of_day_3']
    filter_comp = [col for col in test_x if col.startswith('parf_of_day')]
    test_x = test_x.drop(filter_comp, axis=1)

    # Create y
    y.loc[y['012'] == 0, '015'] = int(0)
    y.loc[y['012'] == 1, '015'] = int(1)
    y.loc[y['012'] == 2, '015'] = int(5)
    y_015 = y[[ARGS.label]].to_numpy().flatten()
    y = y[['012', ARGS.label]].to_numpy()#.flatten()

    # Create qid
    qid = x[['srch_id']]#.to_numpy().flatten()

    # Create x
    used_features = [c for c in list(x) if len(x[c].unique()) > 1]
    x = x[used_features]
    #x = normalize_price_per_query(x)
    x = remove_features(x, ARGS.remove_features)
    col_names = x.columns
    x = x.to_numpy()



    if ARGS.normalize == True:
        x = normalize(x)

    # Split into training and validation set
    train_size = int(x.shape[0]*0.8)
    train_x = x[:train_size]
    train_y = y[:train_size]#[:,1].flatten()
    train_qid = qid[:train_size]
    val_x = x[train_size:]
    val_y = y[train_size:][:,1].flatten()
    val_y_015 = y_015[train_size:]
    val_qid = qid[train_size:]


    # Filter rows
    train_x, train_y, train_qid = keep_rand_rankings(train_x, train_y, train_qid, col_names)
    #train_x, train_y, train_qid = remove_missing_value_rows(train_x, train_y, train_qid, col_names)

    # Upsample clicks and bookings
    if ARGS.upsample == True:
        train_x, train_y, train_qid = upsample(train_x, train_y, train_qid, col_names)

    train_y = train_y[:,1].flatten()

    #Remove outliers
    if ARGS.outlier_reduction == True:
        train_x, train_y, train_qid = remove_outliers(train_x, train_y, train_qid)


    # Count number of entries per query
    if ARGS.upsample == True:
        train_qid_count = train_qid.value_counts()
    else:
        train_qid_count = train_qid['srch_id'].value_counts()

    train_qid_count = pd.DataFrame([train_qid_count]).T.sort_index().to_numpy().flatten()
    val_qid_count = val_qid["srch_id"].value_counts()
    val_qid_count = pd.DataFrame([val_qid_count]).T.sort_index().to_numpy().flatten()


    # Create test data x
    test_qid = 0
    test_prop_id = 0
    if ARGS.test == True:
        test_qid = test_x[['srch_id']].copy().to_numpy().flatten()
        test_prop_id = test_x[['prop_id']].copy().to_numpy().flatten()
        test_x = test_x[used_features]
        #test_x = normalize_price_per_query(test_x)
        test_x = remove_features(test_x, ARGS.remove_features)
        test_x = test_x.to_numpy()
        if ARGS.normalize == True:
                test_x = normalize(test_x)

    return(train_x, train_y, train_qid_count, val_x, val_y, val_qid_count, test_x, test_qid, test_prop_id, val_qid, val_y_015)


def train_model(train_x, train_y, train_qid_count, val_x, val_y, val_qid_count, val_qid, val_y_015):
    print("training model...")
    model = XGBRanker(objective="rank:pairwise", max_depth=3, n_estimators=100)
    model.fit(train_x, train_y, train_qid_count, eval_set=[(val_x, val_y)], eval_group=[val_qid_count], verbose=True)

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





def main():
    # Load data
    y = pd.read_pickle('../data/all_y.pkl')#[:1000000]
    x = pd.read_pickle('../data/all_x.pkl')#[:1000000]
    mpls = pd.read_pickle('../data/all_train_missing_prop_location_score2.pkl')#[:1000000]
    #n = pd.read_pickle('../data/all_nans_train.pkl')#[:1000000]
    #n = n.fillna(np.nan)
    #n_cols = ['visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_starrating',
     #  'prop_review_score', 'srch_query_affinity_score']
    #n = n[n_cols]


    #x = remove_features(x, n_cols)
    x = pd.concat([x, mpls], axis=1)

    test_x = pd.read_pickle('../data/all_test.pkl')#[0:1000]
    test_mpls = pd.read_pickle('../data/all_train_missing_prop_location_score2.pkl')#[:1000]
    #test_n = pd.read_pickle('../data/all_nans_train.pkl')#[:1000000]
    #test_n = test_n.fillna(np.nan)
    #test_n = test_n[n_cols]
    #test_x = remove_features(test_x, n_cols)
    test_x = pd.concat([test_x, test_mpls], axis=1)


    # Prepare data
    train_x, train_y, train_qid_count, val_x, val_y, val_qid_count, test_x, test_qid, test_prop_id, val_qid, val_y_015 = prepare_data(x, y, test_x)

    # Train model
    model = train_model(train_x, train_y, train_qid, train_qid_count, val_x, val_y, val_qid_count, val_qid, val_y_015)

    # Run model on test set
    if ARGS.test == True:
        predict_test_set(model, test_x, test_qid, test_prop_id)





if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--label', default='015', type=str,
                        help='prediction target')
    PARSER.add_argument('--normalize', default=False, type=bool,
                        help='normalize features')
    PARSER.add_argument('--remove_features', default=['prop_id', 'srch_id'], type=list,
                        help='features to be removed before training')
    PARSER.add_argument('--test', default=True, type=bool,
                        help='predict test set and create ranking')
    PARSER.add_argument('--outlier_reduction', default=False, type=bool,
                        help='Remove outlier entries')
    PARSER.add_argument('--z', default=3, type=int,
                        help='z-value threshold for outlier reduction')
    PARSER.add_argument('--n_outlier_features', default=3, type=int,
                        help='max number of features containing outliers')
    PARSER.add_argument('--upsample', default=False, type=bool,
                        help='usample clicks and bookings')
    PARSER.add_argument('--upsample_ratio', default='111', type=str,
                        help='upsample clicks and bookings')




    ARGS = PARSER.parse_args()

    main()
