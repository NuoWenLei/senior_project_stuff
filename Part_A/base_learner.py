import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

def get_base_learner(neurons, layers, pred_size):
	m = Sequential()

	for l in range(layers):
		m.add(Dense(neurons, activation = "relu", name = f"base_model_{l}"))
	
	m.add(Dense(pred_size, activation = "softmax", name = "base_model_output"))

	m.compile(
		loss = "categorical_crossentropy",
		optimizer = "adam",
		metrics = ["accuracy"]
	)

	return m



