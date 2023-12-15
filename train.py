import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error

from xgboost import XGBRegressor


# parameters
n_est = 180  # n_estimators
depth = 5    # max_depth
lr = 0.2    # min_samples_leaf
n_splits = 5
output_file = 'xgb_model.bin'

# Data preparation

df=pd.read_csv('mechanical_properties_low_alloy_steels.csv')

df.columns = df.columns.str.lower().str.lstrip()
df.columns = df.columns.str.replace('0.2% ', '', regex=True)
df.columns = df.columns.str.replace('%', 'perc', regex=True)
df.columns = df.columns.str.replace('°c', 'celcius', regex=True)
df.columns = df.columns.str.replace('+', 'and')
df.columns = df.columns.str.replace('(', '')
df.columns = df.columns.str.replace(')', '')
df.columns = df.columns.str.replace(' ', '_')

del df['alloy_code']
del df['nb_and_ta']
del df['ceq']

tensile_strength_outlier = df[df['tensile_strength_mpa']>1000]
ind_outlier = tensile_strength_outlier.index
df.drop(ind_outlier, inplace=True)

features = ['c', 'si', 'mn', 'p', 's', 'ni', 'cr', 'mo', 'cu', 'v', 'al', 'n', 'temperature_celcius']
targets = ['proof_stress_mpa', 'tensile_strength_mpa', 'elongation_perc', 'reduction_in_area_perc']


# Define metric

def mape_metric(y_val, y_pred):
    dict_mape = {}
    for i in range(4):
        dict_mape[targets[i]] = mean_absolute_percentage_error(y_val[:, i], y_pred[:, i])
    mean_mape = np.mean(list(dict_mape.values()))
    return mean_mape

# validation

print(f'doing validation')

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

model= XGBRegressor(n_estimators=n_est, max_depth=depth, eta=lr, random_state=42)

scores = []

fold = 0

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train[targets].values
    y_val = df_val[targets].values

    model.fit(df_train[features], y_train)
    y_pred = model.predict(df_val[features])
    mape= mape_metric(y_val, y_pred)
    scores.append(mape)

    print(f'mape on fold {fold} is {mape}')
    fold = fold + 1

print('validation results:')
print('mape = %.3f +- %.3f' % (np.mean(scores), np.std(scores)))    

# Training the final model

print('training the final model')


df_full_train = df_full_train.reset_index(drop=True)
y_full_train = df_full_train[targets].values
df_full_train.drop(targets, axis=1, inplace=True)

df_test = df_test.reset_index(drop=True)
y_test = df_test[targets].values
df_test.drop(targets, axis=1, inplace=True)

model.fit(df_full_train, y_full_train)
y_pred = model.predict(df_test)
mape= mape_metric(y_test, y_pred)
print('mape = %.3f' % (mape))

# Save the model

with open(output_file, 'wb') as f_out:
    pickle.dump(model, f_out)

print(f'the model is saved to {output_file}')