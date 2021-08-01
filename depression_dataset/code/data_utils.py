"""
Functions needed for processing the data.
"""
import os
import sys
import json
import pandas as pd
import numpy as np
import pickle as pkl
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelBinarizer
import math
from tqdm import tqdm
from statsmodels.stats.weightstats import DescrStatsW
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression
from sklearn.ensemble import ExtraTreesClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


def data_preprocessing(ARGS):
    '''
    Creates and saves a preprocessed training and test set for the standard
    model with current settings.
    '''

    _part_1(ARGS)
    x_basic, x_temporal, y = _part_2(ARGS)

    return x_basic, x_temporal, y



def _part_1(ARGS, part_of_day=True):
    '''
    Converts the raw dataset with hourly entries to a dataset with daily entries
    '''

    print("--- initial data preprocessing ---")

    # Load initial data
    df = pd.read_csv('data/dataset_mood_smartphone.csv')  # load raw data
    df = df.drop(df.columns[0], axis=1)  # remove first column
    df['time'] = pd.to_datetime(df['time'], infer_datetime_format=True)
    df['date'] = df['time'].dt.date  # reduce time to only date

    if part_of_day:
        df['time'] = df['time'].dt.hour  # reduce date to only date

        # Convert time to part of day
        df.loc[df['time'].isin(range(0, 6)), 'night'] = 1  # [0, 1, 2, 3, 4, 5]
        df.loc[df['time'].isin(range(6, 12)), 'morning'] = 1  # [7, 8, 9, 10, 11]
        df.loc[df['time'].isin(range(12, 18)), 'afternoon'] = 1  # [12, 13, 14, 15, 16, 17]
        df.loc[df['time'].isin(range(18, 24)), 'evening'] = 1  # [18, 19, 20, 21, 22, 23]
        df = df.fillna(0)



        # aggregate part of day to days, using count
        df_pod = pd.pivot_table(df, values=['night', 'morning', 'afternoon', 'evening'], index=['id', 'date'],
                                columns=['variable'], aggfunc=['count'], fill_value=0)

        # aggregate attributes to days, using both sum and mean
        df_attr = pd.pivot_table(df, values='value', index=['id', 'date'],
                                columns=['variable'], aggfunc=[np.sum, np.mean], fill_value=np.nan)


        # Remove the summed columns for the variables mood, circumplex.arousal
        # and circumplex.valce and the averaged columns for all other variables
        for variable in list(df_attr['sum'].columns):
            if variable in ['mood', 'circumplex.arousal', 'circumplex.valence']:
                df_attr.pop(('sum', variable))
            else:
                df_attr.pop(('mean', variable))

        # Reset shape, set column names and merge
        df_attr.columns = list(df_attr['sum'].columns) + list(df_attr['mean'].columns)
        df_pod.columns = list(df_pod['count'].columns)

        df_pod.rename(columns='_'.join, inplace=True)
        df = pd.merge(df_attr, df_pod, on=['date', 'id'])
    else:
        df = pd.pivot_table(df, values='value', index=['id', 'date'],
                                columns=['variable'], aggfunc=[np.sum, np.mean], fill_value=np.nan)

        # Remove the summed columns for the variables mood, circumplex.arousal
        # and circumplex.valce and the averaged columns for all other variables
        for variable in list(df['sum'].columns):
            if variable in ['mood', 'circumplex.arousal', 'circumplex.valence']:
                df.pop(('sum', variable))
            else:
                df.pop(('mean', variable))

        # Reset shape, set column names and merge
        df.columns = list(df['sum'].columns) + list(df['mean'].columns)

    df.reset_index(inplace=True)
    df.index.name = None

    # adding features containing information if variable was missing from entry or not
    # for occuring value, set 1, for missing value, set 0
    df_not_missing = df.notnull()
    df_not_missing = df_not_missing.drop(["id", 'date'], axis=1)
    df_not_missing = df_not_missing*1
    df_not_missing.columns = ['missing_' + str(col) for col in df_not_missing.columns]
    df_not_missing = df_not_missing.loc[:,~df_not_missing.columns.str.startswith('missing_afternoon')] #remove part of day missing features
    df_not_missing = df_not_missing.loc[:,~df_not_missing.columns.str.startswith('missing_evening')] #remove part of day missing features
    df_not_missing = df_not_missing.loc[:,~df_not_missing.columns.str.startswith('missing_morning')] #remove part of day missing features
    df_not_missing = df_not_missing.loc[:,~df_not_missing.columns.str.startswith('missing_night')] #remove part of day missing features

    # df_not_missing = df_not_missing.rename(columns='_'.join, inplace=True)

    # todo: hyperparameter use previous days, or the average of the user
    attr_map = {
        "activity": [],
        "appCat.builtin": [],
        "appCat.communication": [],
        "appCat.entertainment": [],
        "appCat.finance": [],
        "appCat.game": [],
        "appCat.office": [],
        "appCat.other": [],
        "appCat.social": [],
        "appCat.travel": [],
        "appCat.unknown": [],
        "appCat.utilities": [],
        "appCat.weather": [],
        "call": [],
        "screen": [],
        "sms": [],
        "circumplex.arousal": [],
        "circumplex.valence": [],
        "mood": [],
    }

    # filling in nan values
    for index, row in df.iterrows():
        df_same_id = df.loc[df['id'] == row['id']]

        col_names = ["appCat.builtin", "appCat.communication", "appCat.entertainment", "appCat.finance", "appCat.game",
                "appCat.office", "appCat.other", "appCat.social", "appCat.travel", "appCat.unknown", "appCat.utilities",
                "appCat.weather", "call", "screen", "sms"]

        for n in col_names:
            if pd.isnull(row[n]):
                attr_map[n].append(0)
            else:
                attr_map[n].append(row[n])

        col_names = ['circumplex.arousal', 'circumplex.valence' , "mood", "activity"]

        for n in col_names:
            if pd.isnull(row[n]):
                if index == 0:
                    average_same_id = df_same_id[n].mean(skipna=True)
                    attr_map[n].append(average_same_id)
                else:
                    nr_diff_user = 0
                    attr_window = 3 if n == "mood" else 2
                    for i in range(len(df['id'][index - attr_window:index])):
                        if list(df['id'][index - attr_window:index])[i] != row['id']:
                            nr_diff_user += 1

                    average = np.mean(
                        df[n][index - (attr_window - nr_diff_user ):index])
                    if math.isnan(average):
                        average_same_id = df_same_id[n].mean(skipna=True)
                        attr_map[n].append(average_same_id)
                    else:
                        attr_map[n].append(average)
            else:
                attr_map[n].append(row[n])


    for n, item in attr_map.items():
        df.drop(n, axis=1, inplace=True)
        df[n] = item

    # Remove outliers, keeping only rows within 3 standar deviations
    constrains = df.select_dtypes(include=[np.number]).apply(lambda x: np.abs(stats.zscore(x)) < ARGS.z_thresh) .all(axis=1)
    df.drop(df.index[~constrains], inplace=True)
    df_not_missing.drop(df_not_missing.index[~constrains], inplace=True)
    df.reset_index(inplace=True)
    df = df[df.columns.drop(list(df.filter(regex='index')))]
    df_not_missing.reset_index(inplace=True)
    df_not_missing = df_not_missing[df_not_missing.columns.drop(list(df_not_missing.filter(regex='index')))]


    # adding id features
    one_hot_id = LabelBinarizer().fit_transform(df.id)
    user_dataframe = pd.DataFrame(one_hot_id)
    user_dataframe.columns = ['user_' + str(col) for col in user_dataframe.columns]

    df = pd.concat([user_dataframe, df, df_not_missing], axis=1, sort=False)
    # Save dataset as pickle
    df.to_pickle(f'data/preprocessed_part_1_{ARGS.suffix}.pkl')
    # df.to_csv(f'data/preprocessed_part_1_{ARGS.suffix}.csv', index=False, header=True)


def weighted_stats(values, pos, window_size):
    w = DescrStatsW(values, weights=[(1/i) for i in list(pos)], ddof=0)
    return(w)


def aggregate_window(args, window):


    if args.wa_weighted is False:
        with open ("data/pod_missing_agg_dict.json", "r") as f:
            pod_missing = json.load(f)

        if args.wa_pod_missing == 'sum' and args.wa_std is False:
            window_merged = window.groupby([True] * len(window)).agg(pod_missing["sum"])

        elif args.wa_pod_missing == 'sum' and args.wa_std is True:
            window_merged = window.groupby([True] * len(window)).agg(pod_missing["sum_std"])

        elif args.wa_pod_missing == 'mean' and args.wa_std is False:
            window_merged = window.groupby([True] * len(window)).agg(pod_missing["mean"])

        elif args.wa_pod_missing == 'mean' and args.wa_std is True:
            window_merged = window.groupby([True] * len(window)).agg(pod_missing["mean_std"])

    if args.wa_weighted is True:
        # with open ("data/pod_missing_agg_dict.json", "r") as f:
        #     pod_missing = json.load(f)

        window_merged = window.groupby([True] * len(window)).agg({
                    "mood": lambda x: weighted_stats(x, window['pos'], window.shape[0]).mean,
                    "circumplex.arousal":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).mean
                    ,
                    "circumplex.valence":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).mean
                    ,
                    "activity":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).mean
                    ,
                    "call":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "screen":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "sms":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "appCat.builtin":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "appCat.communication":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "appCat.entertainment":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "appCat.finance":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "appCat.game":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "appCat.office":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "appCat.other":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "appCat.social":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "appCat.travel":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "appCat.unknown":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "appCat.utilities":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "appCat.weather":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "afternoon_activity":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "afternoon_appCat.builtin":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "afternoon_appCat.communication":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "afternoon_appCat.entertainment":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "afternoon_appCat.finance":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "afternoon_appCat.game":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "afternoon_appCat.office":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "afternoon_appCat.other":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "afternoon_appCat.social":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "afternoon_appCat.travel":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "afternoon_appCat.unknown":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "afternoon_appCat.utilities":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "afternoon_appCat.weather":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "afternoon_call":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "afternoon_circumplex.arousal":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "afternoon_circumplex.valence":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "afternoon_mood":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "afternoon_screen":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "afternoon_sms":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "evening_activity":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "evening_appCat.builtin":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "evening_appCat.communication":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "evening_appCat.entertainment":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "evening_appCat.finance":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "evening_appCat.game":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "evening_appCat.office":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "evening_appCat.other":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "evening_appCat.social":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "evening_appCat.travel":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "evening_appCat.unknown":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "evening_appCat.utilities":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "evening_appCat.weather":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "evening_call":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "evening_circumplex.arousal":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "evening_circumplex.valence":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "evening_mood":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "evening_screen":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "evening_sms":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "morning_activity":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "morning_appCat.builtin":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "morning_appCat.communication":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "morning_appCat.entertainment":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "morning_appCat.finance":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "morning_appCat.game":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "morning_appCat.office":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "morning_appCat.other":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "morning_appCat.social":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "morning_appCat.travel":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "morning_appCat.unknown":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "morning_appCat.utilities":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "morning_appCat.weather":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "morning_call":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "morning_circumplex.arousal":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "morning_circumplex.valence":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "morning_mood":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "morning_screen":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "morning_sms":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "night_activity":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "night_appCat.builtin":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "night_appCat.communication":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "night_appCat.entertainment":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "night_appCat.finance":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "night_appCat.game":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "night_appCat.office":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "night_appCat.other":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "night_appCat.social":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "night_appCat.travel":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "night_appCat.unknown":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "night_appCat.utilities":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "night_appCat.weather":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "night_call":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "night_circumplex.arousal":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "night_circumplex.valence":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "night_mood":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "night_screen":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "night_sms":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "missing_mood":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "missing_circumplex.arousal":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "missing_circumplex.valence":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "missing_activity":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "missing_call":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "missing_screen":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "missing_sms":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "missing_appCat.builtin":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "missing_appCat.communication":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "missing_appCat.entertainment":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "missing_appCat.finance":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "missing_appCat.game":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "missing_appCat.office":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "missing_appCat.other":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "missing_appCat.social":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "missing_appCat.travel":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "missing_appCat.unknown":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "missing_appCat.utilities":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum
                    ,
                    "missing_appCat.weather":
                        lambda x: weighted_stats(x, window['pos'], window.shape[0]).sum

        })

    window_merged['id'] = window['id'].iloc[0]
    return window_merged


def remove_features(df, ARGS):

    df = df[df.columns.drop(list(df.filter(regex='date')))]
    df = df[df.columns.drop(list(df.filter(regex='id')))]

    if ARGS.use_missing_features == False:
        df = df[df.columns.drop(list(df.filter(regex='missing')))]
    if ARGS.use_pod_features == False:
        df = df[df.columns.drop(list(df.filter(regex='morning')))]
        df = df[df.columns.drop(list(df.filter(regex='night')))]
        df = df[df.columns.drop(list(df.filter(regex='evening')))]
        df = df[df.columns.drop(list(df.filter(regex='afternoon')))]

    for feature in ARGS.remove_standard_features:
        df = df[df.columns.drop(list(df.filter(regex=feature)))]
    return(df)


def normalize(df):
    '''
    Normalize all features expect for id (because it is a string)
    and mood (because less interpretable)
    '''

    result = df.copy()
    for feature in df.columns:
        if feature not in ['id', 'mood', 'date']:
            # if feature == 'circumplex.valence':
            #     print(np.sort(list(df[feature])))
            max_value = df[feature].max()
            min_value = df[feature].min()

            result[feature] = (df[feature] - min_value) / (max_value - min_value)
        else:
            result[feature] = df[feature]
    return result


def _part_2(ARGS):
    '''
    Aggregates windows into single entries with the next day mood as
    corresponding label
    '''

    print("--- data preprocessing for standard model ---")

    df = pd.read_pickle(f'data/preprocessed_part_1_{ARGS.suffix}.pkl')

    y = []
    x_basic = []
    x_temporal = []

    for i in tqdm(range(ARGS.window_size, len(df) - 1)):
        if df.loc[i]["missing_mood"] == 0:
            continue
        df_window = df.loc[i - ARGS.window_size:i - 1, :]
        df_window_date = df.loc[i - ARGS.window_size:i, ['date']]
        user_id = df.loc[i]['id']
        # removing the rows if not from the same user_id
        for j in range(i - ARGS.window_size, i):
            if not np.array_equal(df_window.loc[j]['id'], user_id):
                df_window = df_window.drop(labels=None, axis=0, index=j, columns=None, level=None, inplace=False,
                                           errors='raise')
                df_window_date = df_window_date.drop(labels=None, axis=0, index=j, columns=None, level=None,
                                                     inplace=False,
                                                     errors='raise')
        if df_window.shape[0] == 0:
            continue
        df_window_date = df_window_date.T.squeeze()

        # remove entries with date that is outside of window range
        dates = df_window_date.tolist()
        dates_diffs = []
        for d in dates:
            date_diff = (dates[-1] - d).days
            dates_diffs.append(date_diff)
        dates_diffs = dates_diffs[:-1]
        start = 0
        for j, diff in enumerate(dates_diffs):
            if diff <= ARGS.window_size:
                start = j
                break
        df_window = df_window[start:]
        pos = dates_diffs[start:]
        df_window['pos'] = pos

        # remove date and id feature from df_window because it is not supported by pytorch
        df_window = df_window[df_window.columns.drop(list(df_window.filter(regex='date')))]



        x_temporal.append(df_window)
        y.append(df.loc[i]["mood"])

        aggregated_window = aggregate_window(ARGS, df_window)
        aggregated_window = aggregated_window.fillna(0)
        user_df = df.loc[i][:29].to_frame().T
        aggregated_window= pd.merge(user_df, aggregated_window, on='id')
        df_basic = remove_features(aggregated_window, ARGS)
        df_basic = df_basic.fillna(0)
        x_basic.append(df_basic)

    # Concat target data
    y = pd.DataFrame(y)

    # Concat and normalize basic model data
    x_basic = pd.concat(x_basic)
    if ARGS.normalize is True:
        x_basic = normalize(x_basic)

    print(x_basic[0:10])

    # Normalize temporal model data
    lens = []
    for d in x_temporal:
        lens.append(len(d))
    df_temporal = pd.concat(x_temporal)
    df_temporal = df_temporal[df_temporal.columns.drop(list(df_temporal.filter(regex='id')))]
    df_temporal = df_temporal[df_temporal.columns.drop(list(df_temporal.filter(regex='date')))]
    if ARGS.normalize is True:
        df_temporal = normalize(df_temporal)
    x_temporal = []
    for l in lens:
      x_temporal.append(df_temporal[0:l])
      df_temporal = df_temporal[l:]

    # save data files
    y.to_pickle(f'data/y_{ARGS.suffix}.pkl')
    x_basic.to_pickle(f'data/x_basic_{ARGS.suffix}.pkl')
    with open(f'data/x_temporal_{ARGS.suffix}.pkl', 'wb') as f:
        pkl.dump(x_temporal, f)

    return x_basic, x_temporal, y


def univariate_selection(x, y, score_func):
    x = x[x.columns.drop(list(x.filter(regex='user')))]
    x = x[x.columns.drop(list(x.filter(regex='id')))]
    x = x[x.columns.drop(list(x.filter(regex='mood')))]
    column_names = []
    for i in x.columns:
        column_names.append(i[0])
    x.columns = column_names

    bestfeatures = SelectKBest(score_func=score_func, k='all')
    fit = bestfeatures.fit(x,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(x.columns)
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['feature','score']  #naming the dataframe columns
    featureScores = featureScores.sort_values(by=['score'], ascending=True)

    with sns.axes_style("white"):
        plt.figure(figsize=(10,10))
        ax = sns.barplot(x='score', y='feature', data=featureScores, palette="RdYlGn")
        Path("results").mkdir(parents=True, exist_ok=True)
        plt.savefig('results/univariate_selection.png')
        plt.savefig('results/univariate_selection.eps')


def correlation_heatmap(x, y):
    x = x[x.columns.drop(list(x.filter(regex='user')))]
    x = x[x.columns.drop(list(x.filter(regex='id')))]
    column_names = []
    for i in x.columns:
        column_names.append(i[0])
    x.columns = column_names

    df = pd.concat([x, y], axis=1, join_axes=[x.index])
    corrmat = df.corr(method='pearson') # Can use different methods, have to look up!!!
    best_features = corrmat.index
    corrmat = corrmat[best_features].corr()
    mask = np.triu(np.ones_like(corrmat, dtype=np.bool))

    # take absolute value
    corrmat = corrmat.apply(lambda i: i.abs() if np.issubdtype(i.dtype, np.number) else i)

    plt.figure(figsize=(12,12))
    with sns.axes_style("white"):
        g=sns.heatmap(corrmat, mask=mask, annot=True,
                  square=True, cmap="RdYlGn", cbar=False)
        Path("results").mkdir(parents=True, exist_ok=True)
        plt.savefig('results/correlation_heatmap.png')


def benchmark_data():
    filepath = 'data/dataset_mood_smartphone.csv'
    df = pd.read_csv(filepath)  # load raw data
    df = df.drop(df.columns[0], axis=1)  # remove first column
    df['time'] = pd.to_datetime(df['time'], infer_datetime_format=True)
    df['date'] = df['time'].dt.date  # reduce time to only date

    df = pd.pivot_table(df, values='value', index=['id', 'date'],
                            columns=['variable'], aggfunc=[np.sum, np.mean], fill_value=np.nan)

    # Remove the summed columns for the variables mood, circumplex.arousal
    # and circumplex.valce and the averaged columns for all other variables
    for variable in list(df['sum'].columns):
        if variable in ['mood', 'circumplex.arousal', 'circumplex.valence']:
            df.pop(('sum', variable))
        else:
            df.pop(('mean', variable))

    # Reset shape, set column names and merge
    df.columns = list(df['sum'].columns) + list(df['mean'].columns)
    df.reset_index(inplace=True)
    df.index.name = None

    df = df[["id", "date", "mood"]]
    df_orig = df.notnull()
    df_orig.rename(columns={"mood": "mood_exists"}, inplace=True)
    # df.drop(["id","date"], axis=1, inplace=True)


    l = list()
    for index, row in df.iterrows():
        df_same_id = df.loc[df['id'] == row['id']]

        if pd.isnull(row["mood"]):
            if index == 0:
                average_same_id = df_same_id["mood"].mean(skipna=True)
                l.append(average_same_id)
            else:
                nr_diff_user = 0
                attr_window = 3
                for i in range(len(df['id'][index - attr_window:index])):
                    if list(df['id'][index - attr_window:index])[i] != row['id']:
                        nr_diff_user += 1

                average = np.mean(
                    df["mood"][index - (attr_window - nr_diff_user ):index])
                if math.isnan(average):
                    average_same_id = df_same_id["mood"].mean(skipna=True)
                    l.append(average_same_id)
                else:
                    l.append(average)
        else:
            l.append(row["mood"])

    df.drop("mood", axis=1, inplace=True)
    df["mood"] = l

    fin = pd.concat([df, df_orig], axis=1, sort=False)

    label_idx = fin[fin["mood_exists"]==True].index.values # returns numpy array
    featrue_idx = label_idx - 1 # this is element-wise substraction

    labels = fin.loc[label_idx]["mood"]
    features = fin.loc[featrue_idx]["mood"]

    return features, labels
