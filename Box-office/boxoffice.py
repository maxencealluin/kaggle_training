import pandas as pd
import math
import re

df = pd.read_csv("train.csv")

# print(df.columns)
# print(df.isnull().sum())
df = df.drop(["id", "belongs_to_collection", "homepage", "tagline", "Keywords",
				"crew", "cast", "status", "imdb_id", "poster_path"], axis = 1);
df = df.drop(["overview", "original_title", "title", "production_companies"], axis = 1);
df["runtime"].fillna(0, inplace = True);
df["genres"].fillna("", inplace = True);
print(df.columns)
print(df.isnull().sum())

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

df = df.drop(["genres"], axis = 1);


# replace 0 budget with average
df['budget'].fillna(df[df['budget'] != 0]['budget'].mean(), inplace = True)

# replace release date with 2 columns, time since release and month


# for val in (df['genres'].values):
# 	search = re.findall("\'name\': \'([^']+)\'", val)
# 	for match in search:
# 		if match not in categories:
# 			categories.append(match)

X_train = df.iloc[:, :-1]
Y_train = df.iloc[:, -1:]
