# Import dependencies from train sequence and other modules

import pandas as pd
import numpy as np
from new_train_sequence_1 import tf, train_test_split, MinMaxScaler

# Defined here to give Algorithmic Search access
def get_norm_matrix(embed):
	"""
	Calculate embedding matrix normalized by vector magnitude of each row.

	Args:
	- embed: embedding matrix

	Output:
	- normalized embedding matrix
	"""
	return embed / np.sqrt((embed ** 2).sum(axis = -1))[..., np.newaxis]

def get_max_magnitude(embed):
	"""
	Find largest magnitude out of embedding matrix.

	Args:
	- embed: embedding matrix

	Output:
	- value of largest magnitude
	"""
	return np.sqrt((embed ** 2).sum(axis = -1)).max()

# Import algorithmic search
from New_Algorithmic_Approach import Cosine_Similarity_Algorithmic_Search
import json


def load_dataset_generator(path, dataset_name = "wine"):
	"""
	Load and preprocess data appropriately based on dataset, then returns a dataset generator

	Args:
	- path: path to dataset
	- (Optional) dataset_name: name of dataset to load

	Output:
	- generator object that generates sets of data
	"""
	if dataset_name == "wine":
		df = pd.read_csv(path, delimiter = ";").drop(["pH", "citric acid"], axis = 1)
		new_data = pd.DataFrame(scale_data(df), columns = df.columns)
		generator = create_dataset_generator(new_data)

		return generator
	elif dataset_name == "salary":
		df = pd.read_csv(path)
		new_data = pd.DataFrame(scale_data(df), columns = df.columns)
		generator = create_dataset_generator(new_data)

		return generator

def create_dataset_generator(data):
	"""
	Creates a generator that provides reshuffled data of the same dataset with different target columns.

	Args:
	- data: Pandas DataFrame of data

	Output:
	- creates a generator that yields:
		- a set of (X_train, X_test, y_train, y_test)
		- target column
	"""
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
	"""
	Applies min-max scale on data

	Args:
	- data: Pandas DataFrame of data

	Output:
	- scaled data in numpy
	"""
	minmax = MinMaxScaler()
	new_data = minmax.fit_transform(data)

	return new_data

def onehot_encode(y):
	"""
	One-hot encodes a 1D numpy array

	Args:
	- y: 1D numpy array

	Output:
	- one-hot encoded numpy array (should be 2D)
	"""
	minval = y.min()
	vals = np.int32(y - minval)
	num_classes = np.unique(vals).shape[0]
	new_y = np.zeros((vals.shape[0], num_classes))
	new_y[np.arange(vals.shape[0], dtype = "int32"), vals] = 1.

	return new_y

def create_flow(X, y, batch_size, mode = "train"):
	"""
	Create a generator that returns batches of data

	Args:
	- X: data features
	- y: data targets
	- batch_size: size of every batch of data
	- (Optional) mode: mode of data generation
		- train: infinite iteration over same dataset
		- test: iterates one time over dataset
	"""

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
	"""
	Clears out all fully zero semantic embeddings and associated vocabs.

	Args:
	- vocab: list of vocab words
	- embed_mat: embedding matrix

	Output:
	- filtered vocabs
	- filtered embedding matrix
	"""
	condition = (embed_mat.sum(axis = -1) == 0.0)

	clean_embed = embed_mat[~condition]
	clean_words = np.array(vocab)[~condition]

	return clean_words, clean_embed

def load_embed_and_dictionary(path_to_words, path_to_embeds):
	"""
	Loads and preprocesses vocabulary and embedding matrix

	Args:
	- path_to_words: path to vocab JSON file
	- path_to_embeds: path to embedding matrix NPY file

	Output:
	- vocab list
	- embedding matrix
	- dictionary that maps every word to a categorical number
	"""
	with open(path_to_words, "r") as word_path:
		vocab = json.load(word_path)
	
	with open(path_to_embeds, "rb") as embed_path:
		embed_mat = np.load(embed_path)

	vocab, embed_mat = clean_vocab_embed(vocab, embed_mat)

	vocab_to_number = dict((w, i) for i, w in enumerate(vocab))
	
	return vocab, embed_mat, vocab_to_number

def cosine_similarity(x, y):
	"""
	Finds cosine similarity between two vectors

	Args:
	- x: vector 1
	- y: vector 2

	Output:
	- cosine similarity between vectors
	"""
	x_norm = tf.norm(x, axis = -1)
	y_norm = tf.norm(y, axis = -1)

	return tf.reduce_sum(x * y, axis = -1)/ (x_norm * y_norm)

def average_embed(words_string, embed_mat, dictionary):
	"""
	Calculates averaged semantic embeddings of words in a phrase

	Args:
	- words_string: string of words
	- embed_mat: embedding matrix
	- dictionary: dictionary that maps word to unique number

	Output:
	- averaged embedding of words in string
	"""
	words = words_string.split(" ")

	return np.mean(np.array([embed_mat[dictionary[w]] for w in words]), axis = 0)

def learner_weight_magnitude(learner, params):
	"""
	Calculates average weight magnitude in first layer of a base learner

	Args:
	- learner: base learner model
	- params: general parameters of projects

	Output:
	- numpy value of average magnitude per neuron
	"""
	return tf.reduce_sum(tf.abs(learner.layers[0].weights[0])).numpy() / float(params["FEATURE_SIZE"])

def expand_to_include_masked_feature(arr, col, axis = 0):
	"""
	Expands numpy array to include the masked feature of dataset

	Args:
	- arr: numpy array to be expanded for masking
	- col: column to be expanded to mask
	- axis: axis of masked column in array

	Output:
	- expanded array with a masked column
	"""
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
	"""
	Calculates covariant cosine similarity matrix between embeddings.

	Args:
	- semantic embeddings of features

	Output:
	- covariant similarity matrix between every combination of features
	"""
	norm_embeds = embeds / tf.norm(embeds, axis = -1)[..., tf.newaxis]
	return np.einsum("ijk,ivk->ijv", norm_embeds, norm_embeds)