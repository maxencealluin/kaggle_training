import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

def load_data(path):
	df = pd.read_csv(path, error_bad_lines=False)
	print(df.describe())
	return df

load_data("X_train.csv")
