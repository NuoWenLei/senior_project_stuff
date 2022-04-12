from train_sequence import tf
import numpy as np

class Cosine_Similarity_Algorithmic_Search():

	def __init__(self, vocab, normed_matrix):
		self.vocab = vocab
		self.norm_embed = normed_matrix

	def __call__(self, feature_embeds, cosines):

		w = np.squeeze((feature_embeds / np.sqrt((feature_embeds ** 2).sum(axis = -1))[..., np.newaxis]))

		sims = np.einsum("ik,jk->ijk", self.norm_embed, w).sum(axis = -1)

		most_similar_words = np.argpartition(np.square(cosines - sims).sum(axis = -1), 3)[0:3]

		return most_similar_words

		




