# from tokenize import _all_string_prefixes
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import *
from train_sequence import tf
# from train_sequence import 


class Part_A(tf.keras.models.Model):

	def __init__(self, heads, query_size, feature_size, batch_size, embedding_size, hidden_size, name = "Part_A"):
		super().__init__(name = name)

		self.mha_1 = tf.keras.layers.MultiHeadAttention(num_heads=heads, key_dim=query_size, attention_axes = 2)

		self.norm = tf.keras.layers.LayerNormalization()

		self.d_model = heads * query_size

		self.feature_size = feature_size

		self.batch_size = batch_size

		self.hidden_size = hidden_size

		self.embedding_size = embedding_size

		self.embed_summarizer = tf.keras.layers.Dense(self.hidden_size, activation = "relu")

		self.final_similarity_predictor = tf.keras.layers.Dense(1, activation = "sigmoid")

		self.weight_initializer = tf.random_uniform_initializer(minval = -.1, maxval = .1)
		self.bias_initializer = tf.random_uniform_initializer(minval = -0.1, maxval = 0.1)

		self.W_pred = tf.Variable(self.weight_initializer((self.hidden_size + 3, 1)))
		self.b_pred = tf.Variable(self.bias_initializer((1,)))

		# self.mha_2 = MultiHeadAttention(num_heads=heads, key_dim=query_size)

	def concat_inputs_and_apply_w_and_b(self, x_inputs):

		new_embeds = tf.matmul(tf.concat(x_inputs, axis = -1), self.W_pred) + self.b_pred

		return new_embeds
	
	def call(self, inputs):
		embeds, weights, biases = inputs["embeds"], inputs["weights"], inputs["biases"]

		# Embeds shape: (batch_size, feature_size, d_model)
		# Weights shape: (batch_size, feature_size)
		# Biases shape: (batch_size, 1)
		# Layer Inputs shape: (batch_size, feature_size)
		# Layer Outputs shape: (batch_size, 1)

		embeds *= tf.math.sqrt(tf.cast(self.d_model, tf.float32)) # shape: (batch_size, feature_size, d_model)

		self_attention_embeds = self.mha_1(embeds, embeds) # shape: (batch_size, feature_size, d_model)

		self_attention = self.norm(self_attention_embeds + embeds)[..., tf.newaxis] # shape: (batch_size, feature_size, d_model)

		paraphrased_embed = self.embed_summarizer(self_attention)[:, :, tf.newaxis, ...]

		expanded_paraphrase = tf.repeat(paraphrased_embed, self.d_model, axis = -2)

		repeated_weights = tf.repeat(weights[..., tf.newaxis], self.d_model, axis = -1)

		expanded_weights = tf.reshape(repeated_weights, (self.batch_size, self.feature_size, self.embedding_size, 1))

		expanded_biases = tf.reshape(tf.repeat(biases, self.d_model * self.feature_size, axis = -1), (self.batch_size, self.feature_size, self.d_model))[..., tf.newaxis]

		new_embeds = self.concat_inputs_and_apply_w_and_b([self_attention, expanded_paraphrase, expanded_weights, expanded_biases])

		sim_res = self.final_similarity_predictor(tf.reshape(new_embeds, (self.batch_size, self.feature_size, self.embedding_size)))

		return sim_res









