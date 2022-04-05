# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.layers import *
# from tensorflow.keras.models import *
from baseline_test_sequence import Sequential, Dense

def get_base_learner(neurons, layers, pred_size):
	m = Sequential()

	for l in range(layers):
		m.add(Dense(neurons, activation = "relu", name = f"base_model_{l}"))
	
	m.add(Dense(pred_size, activation = "linear", name = "base_model_output"))

	return m



