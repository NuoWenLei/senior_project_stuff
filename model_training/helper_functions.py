import tensorflow as tf
import numpy as np
from tqdm import tqdm
import json

def load_dictionary(path_to_words, path_to_embed = None):
	with open(path_to_words, "r") as words_json:
		words = json.load(words_json)
	

	if path_to_embed:
		with open(path_to_embed, "rb") as embed_npy:
			embed = np.load(embed_npy)
		
		(clean_words, e) = filter_non_embed_words(words, embed)

		word_dict = dict((w, i) for i, w in enumerate(clean_words))
		return word_dict, clean_words, e
	
	word_dict = dict((w, i) for i, w in enumerate(words))
	
	return word_dict, words


	


def filter_non_embed_words(words, embed):
	condition = (embed.sum(axis = -1) == 0.0)
	return np.array([w for w, d in words])[~condition], embed[~condition]





def generate_word_sets_from_tokens_and_embeddings(token_list, cycles = 5, num_per_set = 13, verbose = 1):

	word_sets = []
	word_targets = []

	token_numbers = [n for n, i in enumerate(token_list)]

	for c in range(cycles):
		if verbose == 1:
			print(f"Cycle {c}")

		word_sets.append(np.random.choice(token_numbers, size = (len(token_numbers), num_per_set)))
		word_targets.extend(token_numbers)

	return np.concatenate(word_sets, axis = 0), np.array(word_targets)





def manual_sample_split(feature_tokens, target_tokens, vocab_size, val = False):
  
	sorted_ft = []
	sorted_tt = []


	for i in range(vocab_size):
		condition = (target_tokens == i)
		sorted_ft.append(feature_tokens[condition])
		sorted_tt.append(target_tokens[condition])

		fts = np.array(sorted_ft, dtype = np.int32)
		tts = np.array(sorted_tt, dtype = np.int32)

		val_set = -2 if val else -1

		X_train_numpy = fts[:, :val_set].reshape(-1, 13)
		y_train_numpy = tts[:, :val_set].reshape(-1)

		p = np.random.permutation(X_train_numpy.shape[0])
		X_train_numpy = X_train_numpy[p]
		y_train_numpy = y_train_numpy[p]

	if val:
		return X_train_numpy, np.squeeze(fts[:, val_set]), np.squeeze(fts[:, -1]), y_train_numpy, np.squeeze(tts[:, val_set]), np.squeeze(tts[:, -1])

	return X_train_numpy, np.squeeze(fts[:, -1]), y_train_numpy, np.squeeze(tts[:, -1])


