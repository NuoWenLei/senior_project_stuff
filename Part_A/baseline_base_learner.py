# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.layers import *
# from tensorflow.keras.models import *
from baseline_test_sequence import tf

def get_base_learner(neurons, layers, pred_size):
	m = tf.keras.models.Sequential()

	for l in range(layers):
		m.add(tf.keras.layers.Dense(neurons, activation = "relu", name = f"base_model_{l}"))
	
	m.add(tf.keras.layers.Dense(pred_size, activation = "linear", name = "base_model_output"))

	return m



