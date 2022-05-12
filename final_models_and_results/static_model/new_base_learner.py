from new_train_sequence_1 import tf
# Import initialized tensorflow from train sequence

def get_base_learner(neurons, layers, pred_size):

	# Simple sequential feed-forward network
	m = tf.keras.models.Sequential()
	
	# Add hidden layers
	for l in range(layers):
		m.add(tf.keras.layers.Dense(neurons, activation = "relu", name = f"base_model_{l}"))
	
	# Add output layer
	m.add(tf.keras.layers.Dense(pred_size, activation = "linear", name = "base_model_output"))

	return m



