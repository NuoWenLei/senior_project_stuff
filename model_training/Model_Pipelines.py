import tensorflow as tf
from tensorflow.keras.models import Model
from Approach_1_Model import *
from Approach_2_Model import *

class Approach_1_Pipeline(Model):

	def __init__(self, query_size, heads, d_model, batch_size, seq_length, d_ff, vocab_size, name = "part_c_and_d"):
		super().__init__(name = name)
		self.query_size = query_size
		self.heads = heads
		self.d_model = d_model
		self.batch_size = batch_size
		self.seq_length = seq_length
		self.d_ff = d_ff
		self.vocab_size = vocab_size

		self.part_c = Part_C(query_size, heads, d_model, batch_size, seq_length, vocab_size)
		self.part_d = Part_D(d_ff, d_model)
		self.part_e = Part_E(query_size, heads, d_model, batch_size, seq_length, d_ff, vocab_size)
		self.final_dense = Dense(self.vocab_size, activation = "softmax")

	def call(self, inputs):
		c_results = self.part_c(inputs)
		d_results = self.part_d(c_results)
		# e_results = self.part_e(d_results)
		reshaped_results = tf.reshape(d_results, (-1, self.d_model))
		prob_res = self.final_dense(reshaped_results)
		return prob_res



class Approach_2_Pipeline(Model):

	def __init__(self, query_size, heads, d_model, batch_size, seq_length, d_ff, vocab_size, embedding_matrix):
		super().__init__(name = "Pretrain_Pipeline")
		self.context_combiner = ContextCombiner(query_size, heads, d_model, batch_size, seq_length, d_ff, vocab_size, embedding_matrix)
		# self.decoder = Decoder(query_size, heads, d_model, batch_size, seq_length, d_ff, vocab_size)

		self.final_dense = Dense(vocab_size, activation = "softmax")

	def call(self, inputs):
		encoded_output = self.context_combiner(inputs)

		# decoded_output = self.decoder(encoded_output)

		final_output = self.final_dense(encoded_output)

		return final_output