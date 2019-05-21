import pandas as pd
import math
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt

import time
from datetime import datetime
now = datetime.now().timestamp()

def standardize(dataframe, col):
	dataframe[col] = (dataframe[col] - dataframe[col].mean()) / dataframe[col].std()

def one_hot_dict(dataframe, col, threshold):
		list = {}
		filtered_list = []
		col_count = str(col) + "_count"
		dataframe[col].fillna("", inplace = True)
		for idx, val in enumerate(dataframe[col].values):
			search = re.findall("\'name\': \'([^']+)\'", val)
			nb = 0
			for match in search:
				nb += 1
				match = match.replace(" ", "")
				if match in list.keys():
					list[match] += 1
				else:
					list[match] = 1
			dataframe.at[idx, col_count] = nb

		standardize(dataframe, col_count)

		for key, val in list.items():
			if val > threshold:
				filtered_list.append(key);
		# print(filtered_list)

		for cat in filtered_list:
			dataframe[cat] = 0

		for idx, val in enumerate(dataframe[col].values):
			search = re.findall("\'name\': \'([^']+)\'", val)
			for match in search:
				match = match.replace(" ", "")
				if match in filtered_list:
					dataframe.at[idx, match] = 1


def prepare_data(df):
	#feature engineering
	df['has_homepage'] = pd.isna(df['homepage'])
	df['has_tagline'] = pd.isna(df['tagline'])
	df['belongs_to_collection'] = pd.isna(df['belongs_to_collection'])
	df = df.drop(["id", "homepage", "tagline",
					"status", "imdb_id", "poster_path"], axis = 1)
	df = df.drop(["overview", "original_title", "title"], axis = 1)
	df["runtime"].fillna(0, inplace = True)
	df["genres"].fillna("", inplace = True)

	# one hot encoding on famous cast + number of cast
	one_hot_dict(df, "cast", 10)
	df = df.drop("cast", axis = 1)

	# one hot encoding on famous crew + number of crew
	one_hot_dict(df, "crew", 10)
	df = df.drop("crew", axis = 1)

	# one hot encoding on genres
	one_hot_dict(df, "genres", 5)
	df = df.drop("genres", axis = 1)

	#one hot encoding on production_countries, spoken_languages, keywords
	one_hot_dict(df, "production_countries", 5)
	one_hot_dict(df, "spoken_languages", 5)
	one_hot_dict(df, "Keywords", 10)
	df = df.drop(["production_countries", "spoken_languages", "Keywords"], axis = 1)


	# one hot encoding on production_companies
	one_hot_dict(df, "production_companies", 10)
	df = df.drop("production_companies", axis = 1)


	# one hot encoding on languages
	dummies = pd.get_dummies(df['original_language'], drop_first = True)
	df = df.drop(['original_language'], axis = 1)
	df = df.join(dummies)

	# replace 0 budget with average and standardize
	mean = round(df[df['budget'] != 0]['budget'].mean())
	df['budget'].fillna(mean, inplace = True)
	df['budget'].replace(0, mean, inplace = True)
	# df['budget'] = df['budget'].apply(lambda x: math.log(x))
	standardize(df, "budget")

	# replace release date with 2 columns, time since release and month
	df['release_date'].fillna("12/31/18", inplace = True)
	df['time_since_release'] = df['release_date'].apply(lambda x: (now - int(datetime.strptime(x, '%m/%d/%y').strftime("%s"))))
	df['time_since_release'] = (df['time_since_release'] - df['time_since_release'].mean()) / df['time_since_release'].std()

	df['month'] = df['release_date'].apply(lambda x: datetime.strptime(x, '%m/%d/%y').month)
	dummies = pd.get_dummies(df['month'], drop_first = True)
	df = df.drop(['release_date'], axis = 1)
	df = df.join(dummies)

	#standardize others columns
	standardize(df, "runtime")
	standardize(df, "popularity")


	# print(df.columns)
	X_train = df.loc[:, df.columns != 'revenue']
	Y_train = df.loc[:, df.columns == 'revenue']
	return X_train, Y_train

df = pd.read_csv("train.csv")
X_train, Y_train = prepare_data(df)

# print(X_train.columns)
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
from sklearn import decomposition
pca = decomposition.PCA(0.95)
pca.fit(X_train)
print(len(X_train.columns))
X_train = pca.transform(X_train)
print(X_train)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 42)

Y_train = np.ravel(np.array(Y_train))

# print(Y_train)


#models

classifiers = {}


#1) Linear regression
from sklearn.linear_model import LinearRegression
classifiers['linear_regression'] = LinearRegression()

from sklearn.linear_model import Lasso
classifiers['Lasso'] = Lasso(max_iter=5000)

#2) regressor forest
from sklearn.ensemble import RandomForestRegressor
classifiers['random_forest'] = RandomForestRegressor(random_state=42, n_estimators=300)

#3) SVR
from sklearn.svm import SVR
classifiers['svr'] = SVR(gamma = 1e-8)

#4) xgboost
from xgboost import XGBRegressor
classifiers['xgboost'] = XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=1, max_depth=12)



#Model evaluation
from math import sqrt, log

#evaluation function
def RMSD(y_pred, y_true):
	 result = 0
	 for pred, truth in zip(y_pred, y_true['revenue']):
		 if (pred > 0 and truth >= 0):
			 diff = log(pred) - log(truth)
			 result += diff * diff
	 return sqrt(result / len(y_pred))

#stacking
X_train_stack = pd.DataFrame()
X_val_stack = pd.DataFrame()
for key, val in classifiers.items():
	print("Training " + key + "...")
	val.fit(X_train, Y_train)
	X_val_stack[key] = val.predict(X_val)
	Y_pred = X_val_stack[key]
	print(key + ": RMSD : " + str(RMSD(Y_pred, Y_val)))
	X_train_stack[key] = val.predict(X_train)

#training and evaluation stacking

model_stack = RandomForestRegressor(random_state=42, n_estimators=300)
model_stack.fit(X_train_stack, Y_train)

Y_pred = model_stack.predict(X_val_stack)
print("stacking : RMSD : " + str(RMSD(Y_pred, Y_val)))


from sklearn.metrics import confusion_matrix
# confusion_mtx = confusion_matrix(Y_val, Y_pred)

# sns.heatmap(confusion_mtx, annot=True,  fmt="d", vmin=0, vmax=20)
# plt.show()

# predict and save submission

# xgb prediction
# results = model_xgb.predict(X_test)

#stacking prediction
X_test_stack = pd.DataFrame()
for key, val in classifiers.items():
	X_test_stack[key] = val.predict(X_test)

results = model_stack.predict(X_test_stack)

import os

if (os.path.exists('submission.txt')):
	 os.remove('submission.txt')
with open('submission.txt', 'w') as file:
	 file.write("id,revenue\n")
	 for i, result in enumerate(results):
		 line = str(i + 3001) + ',' + str(result) + '\n'
		 file.write(line)
	 print("Write successfull")
