import tensorflow as tf
import numpy as np

class Cosine_Similarity_Algorithmic_Search():

	def __init__(self, vocab, normed_matrix):
		self.vocab = vocab
		self.norm_embed = normed_matrix

	def call(self, words, cosines):

		w = self.norm_embed[words] # shape: (word_size, embed_size)

		sims = np.einsum("ik,jk->ijk", self.norm_embed, w).sum(axis = -1)

		most_similar_word = np.argmin(np.math.pow(cosines - sims, 2).sum(axis = -1))

		return most_similar_word




