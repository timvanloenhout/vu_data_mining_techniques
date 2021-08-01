
import pandas as pd
import pickle as pkl
from sklearn.preprocessing import LabelBinarizer


df = pd.read_csv('training_set_VU_DM.csv')  # load raw data
df_test = pd.read_csv('test_set_VU_DM.csv')  # load raw data


#TODO: for the moment removed time
# creating separate features for data and time
# df['date_time'] = pd.to_datetime(df['date_time'], infer_datetime_format=True)
# df['date'] = df['date_time'].dt.date
# df['hour'] = df['date_time'].dt.hour
df = df.drop(columns='date_time')

# one hot encode visitor_loc_country_id
df_visitor_location_country_id= pd.concat([df.visitor_location_country_id, df_test.visitor_location_country_id])
one_hot_id = LabelBinarizer().fit_transform(df_visitor_location_country_id)
one_hot_id = one_hot_id[:len(df)]
df_visitor_loc_country_id = pd.DataFrame(one_hot_id)
df_visitor_loc_country_id.columns = ['visitor_loc_country_' + str(col) for col in df_visitor_loc_country_id.columns]
df = df.drop(columns='visitor_location_country_id')


# one hot encode prop_country_id
df_prop_country_id= pd.concat([df.prop_country_id, df_test.prop_country_id])
one_hot_id = LabelBinarizer().fit_transform(df_prop_country_id)
one_hot_id = one_hot_id[:len(df)]
df_prop_country_id = pd.DataFrame(one_hot_id)
df_prop_country_id.columns = ['prop_country_id_' + str(col) for col in df_prop_country_id.columns]
df = df.drop(columns='prop_country_id')
# df = df.drop(columns='prop_id')

# drop srch_destination_id
df = df.drop(columns='srch_destination_id')
df = df.drop(columns='site_id')
df = df.drop(columns='gross_bookings_usd')
df = df.drop(columns='prop_location_score2')
df = df.drop(columns='prop_log_historical_price')
df = df.drop(columns='orig_destination_distance')

# dropping al the comp features
filter_comp = [col for col in df if col.startswith('comp')]
df = df.drop(filter_comp, axis=1)

# new features containing information about nan values for features
# 'visitor_hist_starrating', 'visitor_hist_adr_usd', 'prop_starrating', 'prop_review_score',
# 'srch_query_affinity_score'
df_missing_visitor_hist_starrating = df['visitor_hist_starrating'].notnull() * 1
df['missing_visitor_hist_starrating'] = df_missing_visitor_hist_starrating

df_missing_visitor_hist_adr_usd = df['visitor_hist_adr_usd'].notnull() * 1
df['missing_visitor_hist_adr_usd'] = df_missing_visitor_hist_adr_usd

df_missing_prop_starrating = df['prop_starrating'].notnull() * 1
df['missing_prop_starrating'] = df_missing_prop_starrating

df_missing_prop_review_score = df['prop_review_score'].notnull() * 1
df['missing_prop_review_score'] = df_missing_prop_review_score

df_missing_srch_query_affinity_score = df['srch_query_affinity_score'].notnull() * 1
df['missing_srch_query_affinity_score'] = df_missing_srch_query_affinity_score

df = df.fillna(df.mean())

with open('train_score2.pkl', 'rb') as f:
    df_score2 = pkl.load(f)

df = pd.concat([df, df_visitor_loc_country_id, df_prop_country_id, df_score2], axis=1, sort=False)

# # Save dataset as pickle
df.to_pickle(f'preprocessed_trainset.pkl')
