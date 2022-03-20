import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from Custom_Layers import *


class Part_C(Model):
  
  def __init__(self, query_size, heads, d_model, batch_size, seq_length, vocab_size, dropout_constant = 0.3, name = "part_c"):
    super().__init__(name = name)

    self.d_model = d_model
    # self.embedding = Embedding(
    #     vocab_size,
    #     d_model,
    #     embeddings_initializer = Constant(embedding_matrix),
    #     trainable = False,
    #     name = f"{name}_embedding"
    # )
    self.embedding = Embedding(
        vocab_size,
        d_model,
        name = f"{name}_embedding"
    )

    self.bias_initializer = tf.random_uniform_initializer(minval = 4., maxval = 5.)

    self.self_attention = MultiHeadAttention(query_size, heads, d_model, batch_size, seq_length, name = "part_c_self_attention")
    self.query_bias = tf.Variable(self.bias_initializer([1, ], dtype = tf.float32))
    self.weight_query_attention = MultiHeadAttention(query_size, heads, d_model, batch_size, seq_length, name = "part_c_query_attention")
    self.norm_1 = LayerNormalization(epsilon = 1e-6)
    self.norm_2 = LayerNormalization(epsilon = 1e-6)

    self.dropout_1 = Dropout(dropout_constant)
    self.dropout_2 = Dropout(dropout_constant)
    self.dropout_3 = Dropout(dropout_constant)

  def call(self, inputs):
    feature_weights, feature_names = inputs["cos_sim"], inputs["tokens"]

    # Get vector embeddings for feature names
    x = self.embedding(feature_names)

    # Normalize by d_model
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

    x = self.dropout_1(x)

    # Self attention inputs, so q, k, and v are all the feature name embeddings
    self_mha_inputs = {
        "query": x,
        "key": x,
        "value": x
    }

    # Apply self multi-head attention layer on the inputs
    self_attention_embeddings = self.self_attention(self_mha_inputs)

    self_attention_embeddings = self.dropout_2(self_attention_embeddings)

    # Add and norm for residual effect
    self_attention_embeddings = self.norm_1(x + self_attention_embeddings)

    # Query multi-head attention inputs
    # so q is the feature weights and the keys and values to query from
    # are the embeddings after self attention
    weighted_embeddings = (feature_weights[..., tf.newaxis] * self_attention_embeddings) + self.query_bias

    query_mha_inputs = {
        "query": weighted_embeddings,
        "key": self_attention_embeddings,
        "value": self_attention_embeddings
    }

    # Apply weight query attention layer on inputs
    weight_queried_attention = self.weight_query_attention(query_mha_inputs)

    weight_queried_attention = self.dropout_3(weight_queried_attention)

    # Add and norm for residual effect
    normed_attention = self.norm_2(weight_queried_attention + tf.reduce_mean(weighted_embeddings, axis = -2)[:, tf.newaxis, :])

    return normed_attention




class Part_D(Model):

  def __init__(self, d_ff, d_model, name = "part_d"):
    super().__init__(name = name)

    self.d_ff = d_ff

    self.feed_forward = FeedForwardNetwork(d_ff, d_model)

    self.norm = LayerNormalization(epsilon = 1e-6)

  def call(self, inputs):

    distributed_context = self.feed_forward(inputs)

    x = self.norm(distributed_context + inputs)
    return tf.reduce_sum(x, axis = -2)[:, tf.newaxis, :]




class Part_E(Model):
	def __init__(self, query_size, heads, d_model, batch_size, seq_length, d_ff, vocab_size, name = "part_e"):
		super().__init__(name = name)

		self.d_model = d_model

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
		attention = self.multi_head_attention_1(mha_inputs)

		attention = self.dropout(attention)

		x = self.norm(attention + inputs)

		y = self.feed_forward(x)

		y = self.norm_2(x + y)

		return y

