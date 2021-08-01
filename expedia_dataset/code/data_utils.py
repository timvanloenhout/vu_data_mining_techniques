import pandas as pd
import math
import numpy as np
from sklearn.utils import resample
import pickle as pkl
import sys
from tqdm import tqdm


def create_labels(df):
    # 012 weighting
    print('012 weighting')
    df['012'] = df.apply(lambda row: row['click_bool'] + row['booking_bool'], axis=1)

    # 158 weighting
    print('158 weighting')
    df['158'] = df.apply(lambda row: 1 + row['click_bool']*4 + row['booking_bool']*3, axis=1)

    # ARP weighting
    print('ARP weighting')
    df['arp'] = df.apply(lambda row: row['012'] * row['position'], axis=1)

    # DCG weighting
    print('DCG weighting')
    df['dcg'] = df.apply(lambda row: row['012'] * math.log2(1+row['position']) , axis=1)

    # Create for for negative information for ARP and DCG
    print('Create for for negative information for ARP and DCG')
    df['-112'] = df.apply(lambda row: -1 + 2*row['click_bool'] + row['booking_bool'], axis=1)
    max_position = df.groupby(['srch_id']).max()['position']
    df['neg_position'] = df.apply(lambda row: max_position[int(row['srch_id'])]-row['position'], axis=1)

    # negative ARP weighting
    print('neg ARP')
    df['neg_arp'] = df.apply(lambda row: (row['-112']*row['position']) if row['-112'] > 0 else (row['-112']*row['neg_position']), axis=1)

    # negative DCG weighting
    print('neg DCG')
    df['neg_dcg'] = df.apply(lambda row: (row['-112']*math.log2(1+row['position'])) if row['-112'] > 0 else (row['-112']*math.log2(1+row['position'])), axis=1)

    df.drop(['position', 'click_bool', 'booking_bool', '-112', 'neg_position'], axis=1, inplace=True)

    return(df)

def upsample_batch(df, ratio):
    # Separate majority and minority classes
    df_0 = df[df['012']==0]
    df_1 = df[df['012']==1]
    df_2 = df[df['012']==2]

    # Upsample minority classes
    if len(df_1) > 0:
        df_1 = resample(df_1, replace=True, n_samples=len(df_0)*int(str(ratio)[1]), random_state=42)
    if len(df_2) > 0:
        df_2= resample(df_2, replace=True, n_samples=len(df_0)*int(str(ratio)[2]), random_state=42)

    # Concatenate majority class with upsampled minority classes
    df_upsampled = pd.concat([df_0, df_1, df_2])
    # print(df['012'].value_counts())
    # print(df_upsampled['012'].value_counts())

    return(df_upsampled)


def create_batches(x, y, label, ratio, type_set): #label for instance ['dcg']
    if type_set == 'train':
        print('--- create train batches ----')
        y = y[['012', label]]
        xy = pd.concat([x, y], axis=1)
        ids = xy['srch_id'].unique()[0:2000] # ONLY 1000 BATCHES
        batches = []

        for i in tqdm(ids):
            batch = xy[:][xy['srch_id'] == i]
            if ratio != False:
                batch = upsample_batch(batch, ratio)
            if label != '012':
                batch.drop(['012'], axis=1, inplace=True)
            batches.append(batch)

        return(batches)
    elif type_set == 'test':
        print('--- create test batches ----')
        ids = x['srch_id'].unique()
        batches = []
        for i in tqdm(ids):
            batch = x[:][x['srch_id'] == i]
            batches.append(batch)
        return(batches)


# # Creating batches train set
# # Load data
# x = pd.read_pickle('data/preprocessed_subset.pkl')
#
# # Split into x and y
# y = x.copy()
# x = x.drop(['position', 'click_bool', 'booking_bool'], axis=1)
# y = y[['srch_id','position', 'click_bool', 'booking_bool']]
#
# # Create relevance labels
# y = create_labels(y)

# # Store preprocessed x and y
# x.to_pickle("data/preprocessed_x.pkl")
# y.to_pickle("data/preprocessed_y.pkl")

y = pd.read_pickle('data/preprocessed_y_all.pkl')
x = pd.read_pickle('data/preprocessed_x_all.pkl')
x = x[x.columns.drop(list(x.filter(regex='visitor_loc_country')))]
x = x[x.columns.drop(list(x.filter(regex='prop_country_id')))]

ratio = False # Or '112' for instance
label = '012'
batches = create_batches(x,y, label, ratio, 'train')
pkl.dump(batches, open( "data/batches_{}_ratio_{}.pkl".format(label, ratio), "wb" ) )



# Creating batches test set
# Load data
# x_test = pd.read_pickle('data/preprocessed_testset.pkl')
#
# # Store preprocessed x and y
# x_test.to_pickle("data/preprocessed_x_test.pkl")
#
# x_test = pd.read_pickle('data/preprocessed_x_test.pkl')
#
# batches = create_batches(x_test, x_test, x_test, x_test, 'test')
# pkl.dump(batches, open( "data/batches_test.pkl", "wb"))
#
# print(batches[0:1])
