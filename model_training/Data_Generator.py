import numpy as np
import tensorflow as tf

class Data_Generator():

	def __init__(self, features, targets, embedding_matrix, vocab_size, batch_size, mode = "train"):
		self.features = features
		self.targets = targets
		self.embedding_matrix = embedding_matrix
		self.vocab_size = vocab_size
		self.mode = mode
		self.batch_size = batch_size

		self.generator = self.create_flow()

	def call(self):
		return next(self.generator)


	def reformat_data_by_tokens(feature_tokens, target_tokens, embedding_matrix, vocab_size):
		f_tokens = []
		f_embeds = []
		f_sims = []
		# t_tokens = []
		# t_embeds = []
		t_onehot = []

		for features, target in zip(feature_tokens, target_tokens):

			f_tokens.append(features)

			t_embed = embedding_matrix[target, ...]

			sim_dots = np.array([np.dot(embedding_matrix[f, ...], t_embed) / (np.linalg.norm(embedding_matrix[f, ...]) * np.linalg.norm(t_embed)) for f in features])

			features[np.isnan(sim_dots)] = target

			f_embeds.append(embedding_matrix[features, ...])

			sim_dots[np.isnan(sim_dots)] = 1.0

			f_sims.append(sim_dots)

			onehot = np.zeros((vocab_size,))
			onehot[target] = 1.

			# t_tokens.append(target)
			# t_embeds.append(embedding_matrix[target, ...])
			t_onehot.append(onehot)

		return {
			"tokens": np.array(f_tokens),
			"embeddings": np.array(f_embeds),
			"cos_sim": np.array(f_sims)
		}, tf.constant(t_onehot, dtype = tf.float32)
		# {
		#     "token": t_tokens,
		#     "embedding": t_embeds,
		#     "onehot": t_onehot
		# }

	
	def create_flow(self):

		if self.mode == "train":

			i = 0

			while True:
			# cos_sim = []
			# embeds = []
			# tokens = []
			
			# batch_targets = []

			

			# for _ in range(batch_size):
			#   cos_sim.append(features[i]["cos_sim"])
			#   embeds.append(features[i]["embeddings"])
			#   tokens.append(features[i]["tokens"])
			#   batch_targets.append(targets[i]["onehot"])

			#   i += 1

			#   if i >= len(targets):
			#     p = np.random.permutation(len(targets))
			#     targets = targets[p]
			#     features = features[p]
			#     i = 0

				i += self.batch_size

				if i >= targets.shape[0]:
					# features, targets = rand_words, rand_targets = generate_word_sets_from_tokens_and_embeddings([w[0] for w in wine_words], wine_embed, cycles = DATA_CYCLES - 1, verbose = 0)
					p = np.random.permutation(len(targets))
					feature_perm = np.random.permutation(features.shape[1])
					targets = targets[p]
					features = features[p][:, feature_perm]
					i = self.batch_size
				
				yield self.reformat_data_by_tokens(features[i-self.batch_size:i, ...], targets[i-self.batch_size:i], self.embedding_matrix, self.vocab_size)
		else:
			i = 0
			while i + self.batch_size < len(targets):
				# cos_sim = []
				# embeds = []
				# tokens = []
				
				# batch_targets = []

				

				# for _ in range(batch_size):
				#   cos_sim.append(features[i]["cos_sim"])
				#   embeds.append(features[i]["embeddings"])
				#   tokens.append(features[i]["tokens"])
				#   batch_targets.append(targets[i]["onehot"])

				#   i += 1

				i += self.batch_size
				
				yield self.reformat_data_by_tokens(features[i-self.batch_size:i, ...], targets[i-self.batch_size:i], self.embedding_matrix, self.vocab_size)
