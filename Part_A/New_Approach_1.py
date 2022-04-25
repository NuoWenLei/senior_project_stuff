# from tokenize import _all_string_prefixes
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import *
from new_train_sequence_1 import tf
# from train_sequence import 

class Part_A_V2(tf.keras.models.Model):

	def __init__(self, batch_size, embedding_size, hidden_size, heads, query_size, feature_size, name = "part_a"):
		super().__init__(name = name)

		self.embedding_size = embedding_size

		self.hidden_size = hidden_size
		
		self.batch_size = batch_size

		self.feature_size = feature_size

		self.mha_1 = tf.keras.layers.MultiHeadAttention(num_heads = heads, key_dim = query_size)

		self.norm_1 = tf.keras.layers.LayerNormalization()

		self.mha_2 = tf.keras.layers.MultiHeadAttention(num_heads = heads, key_dim = query_size)

		self.norm_2 = tf.keras.layers.LayerNormalization()

		self.weight_grad_mixer = tf.keras.layers.Dense(1, activation = "relu")

		self.axis_3_hidden_layers = [
			tf.keras.layers.Dense(self.hidden_size, activation = "relu")
		]

		self.axis_2_hidden_layers = [
			tf.keras.layers.Dense(self.hidden_size, activation = "relu")
		]

		self.final_predictor_layer = tf.keras.layers.Dense(self.feature_size + 1, activation = "linear")

	
	def call(self, inputs):

		embeds, raw_weights, biases, raw_weight_gradients, bias_gradients = inputs["embeds"], inputs["weights"], inputs["biases"], inputs["weight_gradients"], inputs["bias_gradients"]

		# TODO: Add these 2 inputs from new train sequence
		
		# target_col = inputs["target_col"]

		co_sim_matrix = inputs["co_sim_matrix"]

		co_sim_matrix = tf.reshape(co_sim_matrix, (self.batch_size, tf.shape(co_sim_matrix)[-1], tf.shape(co_sim_matrix)[-1]))

		embeds = tf.reshape(embeds, (self.batch_size, -1, self.embedding_size))

		raw_weights = tf.reshape(raw_weights, (self.batch_size, -1))

		raw_weight_gradients = tf.reshape(raw_weight_gradients, (self.batch_size, -1))

		# co_sim_matrix[:, target_col, :] *= 0.0

		# co_sim_matrix[:, :, target_col] *= 0.0

		# embeds[:, target_col, ...] *= 0.0

		# raw_weights[:, target_col] *= 0.0

		# raw_weight_gradients[:, target_col] *= 0.0



		weights = raw_weights / tf.reduce_sum(raw_weights)

		weight_gradients = raw_weight_gradients / tf.reduce_sum(raw_weight_gradients)

		expanded_weights = tf.reshape(weights, (self.batch_size, -1, 1))

		expanded_weight_gradients = tf.reshape(weight_gradients, (self.batch_size, -1, 1))

		expanded_matrix = tf.reshape(co_sim_matrix, (self.batch_size, tf.shape(co_sim_matrix)[1], tf.shape(co_sim_matrix)[2]))

		weight_grad_concat = tf.concat([expanded_weights, expanded_weight_gradients, expanded_matrix], axis = -1)

		weight_grad_mix = self.weight_grad_mixer(weight_grad_concat)



		self_attention_embeds = self.mha_1(embeds, embeds)

		self_attention = self.norm_1(embeds + self_attention_embeds)


		expanded_self_attention = tf.reshape(self_attention, (self.batch_size, -1, self.embedding_size))

		expanded_weight_grad_concat = tf.repeat(weight_grad_mix, self.embedding_size, axis = -1)

		weighted_attention_embeds = self.mha_2(expanded_weight_grad_concat, expanded_self_attention)

		weighted_attention = self.norm_2(weighted_attention_embeds + expanded_self_attention)

		

		axis_3_flattened_attention = tf.reshape(weighted_attention, (self.batch_size, -1, tf.shape(weighted_attention)[-2] * self.embedding_size))

		for l in self.axis_3_hidden_layers:
			axis_3_flattened_attention = l(axis_3_flattened_attention)

		axis_2_flattened_attention = tf.reshape(axis_3_flattened_attention, (self.batch_size, -1))

		for l in self.axis_2_hidden_layers:
			axis_2_flattened_attention = l(axis_2_flattened_attention)

		final_linear_predicts = self.final_predictor_layer(axis_2_flattened_attention)

		return tf.cos(final_linear_predicts)














		



