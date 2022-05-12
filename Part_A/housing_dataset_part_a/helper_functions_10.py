import pandas as pd
import numpy as np
from train_sequence_approach_10 import tf, train_test_split, MinMaxScaler

# Defined here to give Algorithmic Search access
def get_norm_matrix(wine_embed):
	return wine_embed / np.sqrt((wine_embed ** 2).sum(axis = -1))[..., np.newaxis]

def get_max_magnitude(wine_embed):
	return np.sqrt((wine_embed ** 2).sum(axis = -1)).max()

from Algorithmic_Search_10 import Cosine_Similarity_Algorithmic_Search
import json
# from sklearn.model_selection import 
# from sklearn.preprocessing import 

def load_dataset_generator(path, dataset_name = "wine"):
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
	cols = data.columns.values
	i = 0
	while True:
		i += 1
		if i > len(cols):
			i = 1
			p = np.random.permutation(cols.shape[0])
			cols = cols[p]
		
		X_ = data.drop(cols[i-1], axis = 1)
		y_ = data[cols[i-1]]

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
	
	# Add <pad> token
	vocab_to_number["<pad>"] = len(vocab_to_number)
	pad_included_embed_mat = np.vstack((embed_mat, np.zeros(embed_mat.shape[1])))
	
	return vocab, pad_included_embed_mat, vocab_to_number, embed_mat

def cosine_similarity(x, y):
	x_norm = tf.norm(x, axis = -1)
	y_norm = tf.norm(y, axis = -1)

	return tf.reduce_sum(x * y, axis = -1)/ (x_norm * y_norm)

def average_embed(words_string, embed_mat, dictionary):
	words = words_string.split(" ")

	return np.mean(np.array([embed_mat[dictionary[w]] for w in words]), axis = 0)

def learner_weight_magnitude(learner, params):
	return tf.reduce_sum(tf.abs(learner.layers[0].weights[0])).numpy() / float(params["FEATURE_SIZE"])

def pad_text(words_string, embed_mat, dictionary, seq_len):
	words = words_string.split(" ")
	if len(words) > seq_len:
		raise Exception(f"Input Sequence Length above {seq_len} with column {words_string}")
	
	pad_length = seq_len - len(words)

	new_seq = words + ["<pad>"] * pad_length

	return np.array([embed_mat[dictionary[i]] for i in new_seq])

def weighted_average_embed(words_string, embed_mat, dictionary):
	words = words_string.split(" ")

	return np.mean(np.array([embed_mat[dictionary[w]] * np.sqrt((embed_mat[dictionary[w]] ** 2).sum()) for w in words]), axis = 0)