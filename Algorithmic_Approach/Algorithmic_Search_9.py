from train_sequence import tf, get_norm_matrix
import numpy as np

class Cosine_Similarity_Algorithmic_Search():

	def __init__(self, vocab, mat, num_closest = 3):
		self.vocab = vocab
		self.mat = mat
		self.norm_embed = get_norm_matrix(self.mat)
		self.num_closest = num_closest

	def __call__(self, feature_embeds, cosines, pred_mag):

		mag_difference = np.abs(np.sqrt((pred_mag ** 2).sum(axis = -1)) - pred_mag)

		w = np.squeeze((feature_embeds / np.sqrt((feature_embeds ** 2).sum(axis = -1))[..., np.newaxis]))

		sims = np.einsum("ik,jk->ijk", self.norm_embed, w).sum(axis = -1)

		most_similar_words = np.argpartition(np.square(cosines - sims).sum(axis = -1) + mag_difference, self.num_closest)[0:self.num_closest]

		return most_similar_words

		




