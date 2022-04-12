# from tokenize import _all_string_prefixes
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import *
from train_sequence import tf
# from train_sequence import 


class Part_A(tf.keras.models.Model):

	def __init__(self, heads, query_size, feature_size, batch_size, embedding_size, hidden_size, name = "Part_A"):
		super().__init__(name = name)

		# self.mha_1 = tf.keras.layers.MultiHeadAttention(num_heads=heads, key_dim=query_size, attention_axes = 2)

		# self.norm = tf.keras.layers.LayerNormalization()

		self.d_model = heads * query_size

		self.feature_size = feature_size

		self.batch_size = batch_size

		self.hidden_size = hidden_size

		self.embedding_size = embedding_size

		self.weights_batch_norm = tf.keras.layers.BatchNormalization(axis = 0)

		# self.embed_summarizer = tf.keras.layers.Dense(self.hidden_size, activation = "relu")

		self.final_similarity_predictor = tf.keras.layers.Dense(1, activation = "linear")

		# self.weight_initializer = tf.random_uniform_initializer(minval = -.1, maxval = .1)
		# self.bias_initializer = tf.random_uniform_initializer(minval = -0.1, maxval = 0.1)

		# self.W_pred = tf.Variable(self.weight_initializer((self.embedding_size + self.feature_size + 1, 1)))
		# self.b_pred = tf.Variable(self.bias_initializer((1,)))

		self.emed_predict_layers = [
			tf.keras.layers.Dense(self.hidden_size, activation = "relu"),
			tf.keras.layers.Dense(self.hidden_size, activation = "relu")
		]

		self.sim_predict_layers = [
			tf.keras.layers.Dense(self.hidden_size, activation = "relu"),
			tf.keras.layers.Dense(self.hidden_size, activation = "relu")
		]

		# self.mha_2 = MultiHeadAttention(num_heads=heads, key_dim=query_size)

	def concat_inputs_and_apply_w_and_b(self, x_inputs):

		new_embeds = tf.matmul(tf.concat(x_inputs, axis = -1), self.W_pred) + self.b_pred

		return new_embeds

	def concat_inputs_and_apply_dense_layer(self, x_inputs):

		new_embeds = tf.concat(x_inputs, axis = -1)

		for l in self.sim_predict_layers:

			new_embeds = l(new_embeds)
		
		return new_embeds

	def apply_embed_dense_layers(self, embeds):
		for l in self.emed_predict_layers:
			embeds = l(embeds)
		
		return embeds
	
	def call(self, inputs):
		embeds, raw_weights, biases = inputs["embeds"], inputs["weights"], inputs["biases"]

		# Embeds shape: (batch_size, feature_size, d_model)
		# Weights shape: (batch_size, feature_size)
		# Biases shape: (batch_size, 1)
		# Layer Inputs shape: (batch_size, feature_size)
		# Layer Outputs shape: (batch_size, 1)

		# embeds *= tf.math.sqrt(tf.cast(self.d_model, tf.float32)) # shape: (batch_size, feature_size, d_model)

		weights = self.weights_batch_norm(raw_weights)

		# self_attention_embeds = self.mha_1(embeds, embeds) # shape: (batch_size, feature_size, d_model)

		# self_attention = self.norm(self_attention_embeds + embeds) # shape: (batch_size, feature_size, d_model)

		processed_embeds = self.apply_embed_dense_layers(embeds)

		# repeated_weights = tf.repeat(weights[tf.newaxis, ...], self.feature_size, axis = -1)

		# expanded_weights = tf.reshape(repeated_weights, (self.batch_size, self.feature_size, -1))

		expanded_biases = tf.reshape(tf.repeat(biases, self.feature_size, axis = -1), (self.batch_size, self.feature_size))[..., tf.newaxis]

		# new_embeds = self.concat_inputs_and_apply_w_and_b([self_attention, expanded_weights, expanded_biases])

		# return tf.sigmoid(new_embeds)

		new_embeds = self.concat_inputs_and_apply_dense_layer([weights[tf.newaxis, ...], expanded_biases])

		combined_embeds = tf.concat([new_embeds, processed_embeds], axis = -1)

		cos_preds = self.final_similarity_predictor(combined_embeds)

		# return cos_preds

		return tf.clip_by_value(cos_preds, 0.001, 0.999)

		









