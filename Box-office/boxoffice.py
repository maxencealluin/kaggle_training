import pandas as pd
import math
import numpy as np
import re

import time
from datetime import datetime
now = datetime.now().timestamp()

def prepare_data(df):
    # print(df.columns)
    df = df.drop(["id", "belongs_to_collection", "homepage", "tagline", "Keywords",
                    "crew", "cast", "status", "imdb_id", "poster_path"], axis = 1)
    df = df.drop(["overview", "original_title", "title", "production_companies"], axis = 1)
    df["runtime"].fillna(0, inplace = True)
    df["genres"].fillna("", inplace = True)

    # one hot encoding on genres
    genres = []
    for val in (df['genres'].values):
        search = re.findall("\'name\': \'([^']+)\'", val)
        for match in search:
            if match not in genres:
                genres.append(match)

    for genre in genres:
        df[genre] = 0

    for idx, val in enumerate(df['genres'].values):
        search = re.findall("\'name\': \'([^']+)\'", val)
        for match in search:
            df.at[idx, match] = 1

    df = df.drop(["genres"], axis = 1)

    # one hot encoding on languages
    dummies = pd.get_dummies(df['original_language'], drop_first = True)
    df = df.drop(['original_language'], axis = 1)
    df = df.join(dummies)

    # replace 0 budget with average and normalize
    mean = round(df[df['budget'] != 0]['budget'].mean())
    df['budget'].fillna(mean, inplace = True)
    df['budget'].replace(0, mean, inplace = True)
    df['budget'] = (df['budget'] - df['budget'].mean()) / df['budget'].std()

    # replace release date with 2 columns, time since release and month
    df['release_date'].fillna("12/31/18", inplace = True)
    df['time_since_release'] = df['release_date'].apply(lambda x: (now - int(datetime.strptime(x, '%m/%d/%y').strftime("%s"))))
    df['time_since_release'] = (df['time_since_release'] - df['time_since_release'].mean()) / df['time_since_release'].std()

    df['month'] = df['release_date'].apply(lambda x: datetime.strptime(x, '%m/%d/%y').month)
    dummies = pd.get_dummies(df['month'], drop_first = True)
    df = df.drop(['release_date'], axis = 1)
    df = df.join(dummies)

    #normalize others columns
    df['runtime'] = (df['runtime'] - df['runtime'].mean()) / df['runtime'].std()
    df['popularity'] = (df['popularity'] - df['popularity'].mean()) / df['popularity'].std()

    #temp drop
    df = df.drop(["production_countries", "spoken_languages"], axis = 1)

    # print(df.columns)

    X_train = df.loc[:, df.columns != 'revenue']
    Y_train = df.loc[:, df.columns == 'revenue']
    return X_train, Y_train

df = pd.read_csv("train.csv")
X_train, Y_train = prepare_data(df)

X_train.columns = X_train.columns.astype(str)
X_train= X_train.sort_index(axis = 1)


df_test = pd.read_csv("test.csv")
X_test, Y_test = prepare_data(df_test)

X_test.columns = X_test.columns.astype(str)
X_test= X_test.sort_index(axis = 1)

# X_test = X_test.reindex(sorted(X_test.columns), axis=1)


for col in X_test.columns:
    if col not in X_train.columns:
        X_test.drop([col], axis = 1, inplace = True)

for col in X_train.columns:
    if col not in X_test.columns:
        X_train.drop([col], axis = 1, inplace = True)

from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state = 42)

Y_train = np.ravel(np.array(Y_train))

# print(Y_train)


#models
#1) regressor tree
from sklearn.ensemble import RandomForestRegressor

model_rf = RandomForestRegressor(random_state=42, n_estimators=300, verbose=1)
model_rf.fit(X_train, Y_train)

#2) xgboost
from xgboost import XGBRegressor

model_xgb = XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)
model_xgb.fit(X_train, Y_train)

#rate model
from math import sqrt, log

def RMSD(y_pred, y_true):
    result = 0
    for pred, truth in zip(y_pred, y_true['revenue']):
        if (pred > 0 and truth >= 0):
            diff = log(pred) - log(truth)
            result += diff * diff
    return sqrt(result / len(y_pred))

Y_pred = model_rf.predict(X_val)
print("Random forest : RMSD : " + str(RMSD(Y_pred, Y_val)))

Y_pred = model_xgb.predict(X_val)
print("XGBoost : RMSD : " + str(RMSD(Y_pred, Y_val)))

from sklearn.metrics import confusion_matrix
# confusion_mtx = confusion_matrix(Y_val, Y_pred)

# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.heatmap(confusion_mtx, annot=True,  fmt="d", vmin=0, vmax=20)
# plt.show()

# predict and save submission

results = model_xgb.predict(X_test)

import os

if (os.path.exists('submission.txt')):
    os.remove('submission.txt')
with open('submission.txt', 'w') as file:
    file.write("id,revenue\n")
    for i, result in enumerate(results):
        line = str(i + 3001) + ',' + str(result) + '\n'
        file.write(line)
    print("Write successfull")
