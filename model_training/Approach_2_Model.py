import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from Custom_Layers import *

class ContextCombiner(Model):

	def __init__(self, query_size, heads, d_model, batch_size, seq_length, d_ff, vocab_size, embedding_matrix, hidden_size = 20, name = "context_combiner"):

		super().__init__(name = name)

		self.hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.d_model = d_model
		self.batch_size = batch_size

		self.embeddings = Embedding(
			self.vocab_size,
			self.d_model,
			name = f"{name}_embedding"
		)

		self.embed_self_attention = MultiHeadAttention(query_size, heads, d_model, batch_size, seq_length, name = f"{name}_self_embed_attention")

		self.dropout_1 = Dropout(0.3)

		# Weight initializers
		self.uniform_initializer = tf.random_uniform_initializer(minval = -0.1, maxval = 0.1)

		# Bias initializers
		self.uniform_initializer_b = tf.random_uniform_initializer(minval = 100., maxval = 150.)

		self.W = tf.Variable(self.uniform_initializer(shape = [hidden_size + 2, 1], dtype = tf.float32))

		self.b = tf.Variable(self.uniform_initializer_b(shape = (1,1), dtype = tf.float32))

		self.embedding_generalizer = Dense(self.hidden_size, activation = "relu")

		self.feed_forward = FeedForwardNetwork(d_ff, d_model, name = f"{name}_ff")

		self.norm = LayerNormalization()

	def repeat_and_concat(self, repeat_along, to_repeats):
		repeateds = [tf.repeat(to_repeat[:, :, tf.newaxis, ...], self.d_model, axis = -2) for to_repeat in to_repeats]
		return tf.concat([repeat_along] + repeateds, axis = -1)

	def call(self, inputs):

		feature_weights, feature_names = inputs["cos_sim"], inputs["tokens"]

		feature_weights = feature_weights[..., tf.newaxis]

		x = self.embeddings(feature_names)

		# Normalize by d_model
		x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

		embeds = self.dropout_1(x)

		mha_inputs = {
			"key": embeds,
			"query": embeds,
			"value": embeds
		}

		self_attention_embeds = self.embed_self_attention(mha_inputs)

		generalized_embeds = self.embedding_generalizer(self_attention_embeds)

		concatted_embeds = self.repeat_and_concat(embeds[..., tf.newaxis], [feature_weights, generalized_embeds])

		weight_applied_embeds = tf.matmul(concatted_embeds, self.W) + self.b

		feature_summed_embeds = tf.reshape(tf.reduce_sum(weight_applied_embeds, axis = -3), (self.batch_size, self.d_model))

		ff_embeds = self.feed_forward(feature_summed_embeds)

		normed_output = self.norm(ff_embeds + feature_summed_embeds)

		return normed_output

class Decoder(Model):
	def __init__(self, query_size, heads, d_model, batch_size, seq_length, d_ff, vocab_size, name = "decoder"):
		super().__init__(name = name)

		self.d_model = d_model
		self.batch_size = batch_size

		self.multi_head_attention_1 = MultiHeadAttention(query_size, heads, d_model, batch_size, seq_length, name = f"{name}_mha_1")

		self.norm = LayerNormalization(epsilon = 1e-6)
		self.norm_2 = LayerNormalization(epsilon = 1e-6)
		self.feed_forward = FeedForwardNetwork(d_ff, d_model, name = f"{name}_ff")

		self.dropout = Dropout(0.3)

		assert d_model == query_size * heads, f"d_model size incorrect: {d_model} != {query_size} * {heads}"

	def call(self, inputs):

		"""
		inputs["prev"] previous word embedding
		inputs["encoder"] encoder outputs
		"""

		mha_inputs = {
			"query": inputs,
			"key": inputs,
			"value": inputs
		}

		# Multi-head attention because q and k are from encoder and v is from self
		attention = tf.reshape(self.multi_head_attention_1(mha_inputs), (self.batch_size, self.d_model))

		attention = self.dropout(attention)

		x = self.norm(attention + inputs)

		y = self.feed_forward(x)

		y = self.norm_2(x + y)

		return y




