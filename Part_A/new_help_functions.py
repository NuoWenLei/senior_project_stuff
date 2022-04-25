import pandas as pd
import numpy as np
from new_train_sequence_1 import tf, train_test_split, MinMaxScaler

# Defined here to give Algorithmic Search access
def get_norm_matrix(wine_embed):
	return wine_embed / np.sqrt((wine_embed ** 2).sum(axis = -1))[..., np.newaxis]

def get_max_magnitude(wine_embed):
	return np.sqrt((wine_embed ** 2).sum(axis = -1)).max()

from New_Algorithmic_Approach import Cosine_Similarity_Algorithmic_Search
import json
# from sklearn.model_selection import 
# from sklearn.preprocessing import 

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
		# 	p = np.random.permutation(cols.shape[0])
		# 	cols = cols[p]
		
		X_ = data.drop(cols[i-1], axis = 1)
		y_ = data[cols[i-1]]

		yield train_test_split(X_, y_, random_state = 0, train_size = .8), i-1

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

def clean_vocab_embed(vocab, embed_mat):
	condition = (embed_mat.sum(axis = -1) == 0.0)

	clean_embed = embed_mat[~condition]
	clean_words = np.array(vocab)[~condition]

	return clean_words, clean_embed

def load_embed_and_dictionary(path_to_words, path_to_embeds):
	with open(path_to_words, "r") as word_path:
		vocab = json.load(word_path)
	
	with open(path_to_embeds, "rb") as embed_path:
		embed_mat = np.load(embed_path)

	vocab, embed_mat = clean_vocab_embed(vocab, embed_mat)

	vocab_to_number = dict((w, i) for i, w in enumerate(vocab))
	
	return vocab, embed_mat, vocab_to_number

def cosine_similarity(x, y):
	x_norm = tf.norm(x, axis = -1)
	y_norm = tf.norm(y, axis = -1)

	return tf.reduce_sum(x * y, axis = -1)/ (x_norm * y_norm)

def average_embed(words_string, embed_mat, dictionary):
	words = words_string.split(" ")

	return np.mean(np.array([embed_mat[dictionary[w]] for w in words]), axis = 0)

def learner_weight_magnitude(learner, params):
	return tf.reduce_sum(tf.abs(learner.layers[0].weights[0])).numpy() / float(params["FEATURE_SIZE"])

def expand_to_include_masked_feature(arr, col, axis = 0):
	dims = [i for i in np.shape(arr)]
	dims[axis] += 1
	expanded_zeros = np.zeros(dims, dtype = np.float32)

	if axis == 0:
		expanded_zeros[:col] = arr[:col]
		expanded_zeros[col+1:] = arr[col:]
	if axis == 1:
		expanded_zeros[:, :col] = arr[:, :col]
		expanded_zeros[:, col+1:] = arr[:, col:]
	if axis == 2:
		expanded_zeros[:, :, :col] = arr[:, :, :col]
		expanded_zeros[:, :, col+1:] = arr[:, :, col:]

	return expanded_zeros

def get_covariance_similarity_matrix(embeds):
	norm_embeds = embeds / tf.norm(embeds, axis = -1)[..., tf.newaxis]
	return np.einsum("ijk,ivk->ijv", norm_embeds, norm_embeds)