# from tokenize import _all_string_prefixes
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import *
from train_sequence_approach_9 import tf
# from train_sequence import 


class Part_A(tf.keras.models.Model):

	def __init__(self, heads, query_size, feature_size, batch_size, embedding_size, hidden_size, max_magnitude, name = "Part_A"):
		super().__init__(name = name)

		self.mha_1 = tf.keras.layers.MultiHeadAttention(num_heads=heads, key_dim=query_size, attention_axes = 2)

		self.norm = tf.keras.layers.LayerNormalization()

		self.d_model = heads * query_size

		self.feature_size = feature_size

		self.batch_size = batch_size

		self.hidden_size = hidden_size

		self.embedding_size = embedding_size

		self.max_magnitude = max_magnitude

		# self.embed_summarizer = tf.keras.layers.Dense(self.hidden_size, activation = "relu")

		self.final_similarity_predictor = tf.keras.layers.Dense(1, activation = "linear")

		# self.weight_initializer = tf.random_uniform_initializer(minval = -.1, maxval = .1)
		# self.bias_initializer = tf.random_uniform_initializer(minval = -0.1, maxval = 0.1)

		# self.W_pred = tf.Variable(self.weight_initializer((self.embedding_size + self.feature_size + 1, 1)))
		# self.b_pred = tf.Variable(self.bias_initializer((1,)))

		self.sim_predict_layers = [
			tf.keras.layers.Dense(self.hidden_size, activation = "relu")
		]

		# self.mha_2 = MultiHeadAttention(num_heads=heads, key_dim=query_size)

	def concat_inputs_and_apply_w_and_b(self, x_inputs):

		new_embeds = tf.matmul(tf.concat(x_inputs, axis = -1), self.W_pred) + self.b_pred

		return new_embeds

	def concat_inputs_and_apply_dense_layer(self, x_inputs):

		new_sim_embeds = tf.concat(x_inputs, axis = -1)

		for l in self.sim_predict_layers:

			new_sim_embeds = l(new_sim_embeds)
		
		return new_sim_embeds
	
	def call(self, inputs):
		embeds, raw_weights, biases, raw_weight_gradients, bias_gradients = inputs["embeds"], inputs["weights"], inputs["biases"], inputs["weight_gradients"], inputs["bias_gradients"]

		# Embeds shape: (batch_size, feature_size, d_model)
		# Weights shape: (batch_size, feature_size)
		# Biases shape: (batch_size, 1)
		# Layer Inputs shape: (batch_size, feature_size)
		# Layer Outputs shape: (batch_size, 1)

		# embeds *= tf.math.sqrt(tf.cast(self.d_model, tf.float32)) # shape: (batch_size, feature_size, d_model)

		weights = raw_weights / tf.reduce_sum(raw_weights) # tf.sqrt(tf.reduce_sum(raw_weights ** 2))

		weight_gradients = raw_weight_gradients / tf.reduce_sum(raw_weight_gradients) #tf.sqrt(tf.reduce_sum(raw_weight_gradients ** 2))

		# bias_gradients = raw_bias_gradients / tf.sqrt(tf.reduce_sum(raw_bias_gradients ** 2))

		self_attention_embeds = self.mha_1(embeds, embeds) # shape: (batch_size, feature_size, d_model)

		self_attention = self.norm(self_attention_embeds + embeds) # shape: (batch_size, feature_size, d_model)

		# repeated_weights = tf.repeat(weights[tf.newaxis, ...], self.feature_size, axis = -1)

		# expanded_weights = tf.reshape(repeated_weights, (self.batch_size, self.feature_size, -1))

		embed_magnitudes = tf.math.sqrt(tf.reduce_sum(embeds ** 2, axis = -1))[..., tf.newaxis] / self.max_magnitude

		# embed_magnitudes = tf.sqrt((embeds ** 2).sum(axis = -1))[..., tf.newaxis]
		
		expanded_weights = weights[tf.newaxis, ...]

		expanded_weight_gradients = weight_gradients[tf.newaxis, ...]

		expanded_biases = tf.reshape(tf.repeat(biases, self.feature_size, axis = -1), (self.batch_size, self.feature_size))[..., tf.newaxis]

		expanded_bias_gradients = tf.reshape(tf.repeat(bias_gradients, self.feature_size, axis = -1), (self.batch_size, self.feature_size))[..., tf.newaxis]

		# new_embeds = self.concat_inputs_and_apply_w_and_b([self_attention, expanded_weights, expanded_biases])

		# return tf.sigmoid(new_embeds)

		new_sim_embeds = self.concat_inputs_and_apply_dense_layer([self_attention, expanded_weights, expanded_weight_gradients, expanded_biases, expanded_bias_gradients, embed_magnitudes])

		cos_preds = self.final_similarity_predictor(new_sim_embeds)

		# mag_preds = self.final_magnitude_predictor(new_mag_embeds)

		return tf.math.cos(cos_preds)

		









