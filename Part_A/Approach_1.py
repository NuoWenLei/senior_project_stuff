# from tokenize import _all_string_prefixes
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import *
from train_sequence import tf
# from train_sequence import 

class MetaLSTMCell(tf.keras.layers.Layer):

	def __init__(self, hidden_size, feature_size, name = "MetaLSTMCell"):

		super().__init__(name = name)

		self.hidden_size = hidden_size
		self.feature_size = feature_size

		self.weight_initializer = tf.random_normal_initializer()
		self.uniform_initializer_bI = tf.random_uniform_initializer(minval = -5., maxval = -4.)
		self.uniform_initializer_bF = tf.random_uniform_initializer(minval = 4., maxval = 5.)

		self.W_i = tf.Variable(self.weight_initializer((self.hidden_size + 4, 1)))
		self.W_f = tf.Variable(self.weight_initializer((self.hidden_size + 4, 1)))

		self.b_i = tf.Variable(self.uniform_initializer_bI((1, 1)))
		self.b_f= tf.Variable(self.uniform_initializer_bF((1, 1)))

	def call(self, inputs, hidden_state):

		prev_f, prev_i, prev_c = hidden_state

		new_self_attention, x_inputs = inputs

		new_f = tf.matmul(tf.concat(x_inputs + [prev_c, prev_f], axis = -1), self.W_f) + self.b_f

		new_i = tf.matmul(tf.concat(x_inputs + [prev_c, prev_i], axis = -1), self.W_i) + self.b_i

		new_c = (tf.math.sigmoid(new_f) * prev_c) - (tf.math.sigmoid(new_i) * tf.reshape(new_self_attention, (-1, 1)))

		return new_c, new_f, new_i



class Part_A(tf.keras.models.Model):

	def __init__(self, heads, query_size, feature_size, batch_size, embedding_size, name = "Part_A"):
		super().__init__(name = name)

		self.mha_1 = tf.keras.layers.MultiHeadAttention(num_heads=heads, key_dim=query_size, attention_axes = 2)

		self.norm = tf.keras.layers.LayerNormalization()

		self.d_model = heads * query_size

		self.feature_size = feature_size

		self.batch_size = batch_size

		self.hidden_size = embedding_size

		self.embedding_size = embedding_size

		self.embed_summarizer = tf.keras.layers.LSTMCell(self.embedding_size, activation = "relu")

		self.final_similarity_predictor = tf.keras.layers.Dense(1, activation = "sigmoid")

		self.meta_lstm = MetaLSTMCell(self.hidden_size, self.feature_size)

		self.prev_embeds = []
		self.prev_I = []
		self.prev_F = []

		# self.mha_2 = MultiHeadAttention(num_heads=heads, key_dim=query_size)
	
	def call(self, inputs, hidden_states = None):
		embeds, weights, biases = inputs["embeds"], inputs["weights"], inputs["biases"]

		# Embeds shape: (batch_size, feature_size, d_model)
		# Weights shape: (batch_size, feature_size)
		# Biases shape: (batch_size, 1)
		# Layer Inputs shape: (batch_size, feature_size)
		# Layer Outputs shape: (batch_size, 1)

		embeds *= tf.math.sqrt(tf.cast(self.d_model, tf.float32)) # shape: (batch_size, feature_size, d_model)

		self_attention_embeds = self.mha_1(embeds, embeds) # shape: (batch_size, feature_size, d_model)

		self_attention = self.norm(self_attention_embeds + embeds) # shape: (batch_size, feature_size, d_model)

		if hidden_states is None:
			h = tf.zeros((self.batch_size * self.feature_size, self.embedding_size), dtype = tf.float32)
			c = tf.zeros((self.batch_size * self.feature_size, self.embedding_size), dtype = tf.float32)
			f = tf.zeros((self.batch_size, self.feature_size, self.embedding_size, 1))
			i = tf.zeros((self.batch_size, self.feature_size, self.embedding_size, 1))
			embed_c = tf.zeros((self.batch_size, self.feature_size, self.embedding_size, 1))
			hidden_states = [(h, c), (f, i, embed_c)]

		_, (hidden_x, cell_x) = self.embed_summarizer(tf.reshape(self_attention, (-1, self.embedding_size)), hidden_states[0])

		paraphrased_embed = tf.reshape(hidden_x, (self.batch_size, self.feature_size, -1))[:, :, tf.newaxis, ...]

		expanded_paraphrase = tf.repeat(paraphrased_embed, self.d_model, axis = -2)

		repeated_weights = tf.repeat(weights[..., tf.newaxis], self.d_model, axis = -1)

		expanded_weights = tf.reshape(repeated_weights, (self.batch_size, self.feature_size, self.embedding_size, 1))

		expanded_biases = tf.reshape(tf.repeat(biases, self.d_model * self.feature_size, axis = -1), (self.batch_size, self.feature_size, self.d_model))[..., tf.newaxis]

		# PERHAPS ADD INPUT AND FORGET GATE?
		# CHECK META-OPTIMIZER TO SEE AT WHAT STAGE THE GATES ARE APPLIED

		new_c, new_f, new_i = self.meta_lstm((self_attention, [expanded_paraphrase, expanded_weights, expanded_biases]), hidden_states[1])
		# previous is previous embeddings
		# new is self attention

		sim_res = self.final_similarity_predictor(new_c)

		return sim_res, [(hidden_x, cell_x), (new_f, new_i, new_c)]









