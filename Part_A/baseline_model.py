import tensorflow as tf
from tensorflow.keras.models import Model
from helper_functions import *


class Baseline_Model(Model):

	def __init__(self, name = "baseline_part_a"):
		super().__init__(name = name)

	def call(self, inputs):
		embeds, weights, biases = inputs["embeds"], inputs["weights"], inputs["biases"]

		scaled_weights = weights / tf.reduce_sum(weights)

		new_embed = tf.reduce_sum(scaled_weights * embeds)

		cos_sims = []

		for e in embeds:
			cos_sims.append(cosine_similarity(new_embed, e))
		
		return tf.constant(cos_sims, dtype = tf.float32)
