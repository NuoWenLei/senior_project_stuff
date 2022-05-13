from train_sequence_approach_10 import tf
from helper_functions_10 import get_norm_matrix, get_max_magnitude
import numpy as np

class Cosine_Similarity_Algorithmic_Search():

	def __init__(self, vocab, mat, num_closest = 3, min_char = 5):
		filtered_vocab, filtered_mat = self.cut_low_character_words(vocab, mat, min_char)
		self.vocab = filtered_vocab
		self.mat = filtered_mat
		self.max_embed = get_max_magnitude(self.mat)
		self.norm_embed = get_norm_matrix(self.mat)
		self.num_closest = num_closest

	def cut_low_character_words(self, vocab, mat, min_char):
		new_voc = []
		new_mat = []
		for i, v in enumerate(vocab):
			if len(v) >= min_char:
				new_voc.append(v)
				new_mat.append(mat[i])
		
		return np.array(new_voc), np.array(new_mat)

	def __call__(self, feature_embeds, cosines, pred_mag = None):

		w = np.squeeze((feature_embeds / np.sqrt((feature_embeds ** 2).sum(axis = -1))[..., np.newaxis]))

		sims = np.einsum("ik,jk->ijk", self.norm_embed, w).sum(axis = -1)

		if pred_mag is not None:

			mag_difference = np.abs((np.sqrt((self.mat ** 2).sum(axis = -1)) / self.max_embed) - pred_mag)

			most_similar_words = np.argpartition(np.square(cosines - sims).sum(axis = -1) + mag_difference, self.num_closest)[0:self.num_closest]
		
		else:

			most_similar_words = np.argpartition(np.square(cosines - sims).sum(axis = -1), self.num_closest)[0:self.num_closest]

		return most_similar_words

		




