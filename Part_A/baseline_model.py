# import tensorflow as tf
# from tensorflow.keras.models import Model
# from helper_functions import *
from baseline_test_sequence import tf, cosine_similarity

class Baseline_Model(tf.keras.models.Model):

	def __init__(self, name = "baseline_part_a"):
		super().__init__(name = name)

	def call(self, inputs):
		embeds, weights, biases = inputs["embeds"], inputs["weights"], inputs["biases"]

		scaled_weights = weights / tf.reduce_sum(weights)

		new_embed = tf.reduce_sum(scaled_weights * embeds, axis = -2)
		
		return cosine_similarity(new_embed, embeds)

