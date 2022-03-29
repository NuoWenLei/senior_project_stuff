import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_data(path):
	df = pd.read_csv(path, delimiter = ";")
	X_ = scale_data(df.drop("quality", axis = 1))
	y_ = onehot_encode(df["quality"])
	dataset = train_test_split(X_, y_, train_size = .8, random_state = 0)

	return dataset

def scale_data(data):
	minmax = MinMaxScaler()
	new_data = minmax.fit_transform(data)

	return new_data

def onehot_encode(y):
	minval = y.min()
	vals = np.int32(y - minval)
	num_classes = np.unique(vals).shape[0]
	new_y = np.zeros((vals.shape[0], num_classes))
	new_y[np.arange(vals.shape[0], dtype = "int32"), vals] = 1.

	return new_y


