import pandas as pd
import numpy as np
import tensorflow as tf
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_dataset_generator(path):
	df = pd.read_csv(path, delimiter = ";").drop(["pH", "citric acid"], axis = 1)
	new_data = pd.DataFrame(scale_data(df), columns = df.columns)
	generator = create_dataset_generator(new_data)

	return generator

def create_dataset_generator(data):
	cols = data.columns.values
	i = 0
	while True:
		i += 1
		if i > len(cols):
			i = 1
			p = np.random.permutation(cols.shape[0])
			cols = cols[p]
		
		X_ = data.drop(cols[i], axis = 1)
		y_ = data[cols[i]]

		yield train_test_split(X_, y_, random_state = 0, train_size = .8)

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

def create_flow(X, y, batch_size, mode = "train"):

	if mode == "train":
		i = 0
		while True:
			i += batch_size
			if i + batch_size >= y.shape[0]:
				i = batch_size
				p = np.random.permutation(y.shape[0])
				X = X[p]
				y = y[p]
			yield X[i-batch_size:i], y[i-batch_size:i]
	
	else:
		i = 0
		while i + batch_size < y.shape[0]:
			i += batch_size
			yield X[i-batch_size:i], y[i-batch_size:i]

def load_embed_and_dictionary(path_to_words, path_to_embeds):
	with open(path_to_words, "r") as word_path:
		vocab = json.load(word_path)
	
	with open(path_to_embeds, "rb") as embed_path:
		embed_mat = np.load(embed_path)

	vocab_to_number = dict((w, i) for i, w in enumerate(vocab))
	
	return vocab, embed_mat, vocab_to_number

def cosine_similarity(x, y):
	x_norm = np.linalg.norm(x)
	y_norm = np.linalg.norm(y)

	return np.dot(x, y) / (x_norm * y_norm)