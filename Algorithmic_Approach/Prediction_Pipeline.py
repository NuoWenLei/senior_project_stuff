from train_sequence import tf, np, tqdm
from helper_functions import *
from base_learner import *

K = tf.keras.backend

# class Interpreted_Learner_Pipeline():

# 	def __init__(self, part_a, algo, params):
# 		self.part_a = part_a
# 		self.algo = algo
# 		self.base_model = get_base_learner(params["BASE_NEURONS"], params["BASE_LAYERS"], params["NUM_CLASSES"])
# 		self.hidden_states = [None]
	
# 	def parse(self, neuron_number):
# 		w = self.base_model.layers[0].weights[0].numpy()[:, neuron_number:neuron_number+1]
# 		b = self.base_model.layers[0].weights[1].numpy()[neuron_number:neuron_number+1]


		
# 		pass
# 	def fit(self, X, y, batch_size, epochs, steps_per_epoch):
		
# 		train_generator = create_flow(X, y, batch_size)

# 		for e in range(epochs):
# 			for s in range(steps_per_epoch):
# 				batch = next(train_generator)

class Train_Pipeline:

	def __init__(self, base_model, part_a, algo, neuron_number, feature_embeds, num_neurons, loss_function, optimizer):
		self.base_model = base_model
		self.algo = algo
		self.part_a = part_a
		self.neuron_number = neuron_number
		self.feature_embeds = feature_embeds
		self.num_neurons = num_neurons
		self.loss_function = loss_function
		self.optimizer = optimizer
		self.interpreter_callback = Interpreter_Custom_Callback(self.base_model, self.part_a, self.algo, self.neuron_number, self.feature_embeds, self.num_neurons)

	def fit(self, X, y, batch_size, epochs):
		steps = X.shape[0] // batch_size
		for epoch in range(epochs):
			print(f"On Epoch {epoch}")
			p = np.random.permutation(X.shape[0])
			X = X[p]
			y = y[p]
			gradients = []
			for step in tqdm(range(steps)):
				X_batch = X[step * batch_size: (step + 1) * batch_size]
				y_batch = y[step * batch_size: (step + 1) * batch_size]
				
				with tf.GradientTape() as gt:
					y_pred = self.base_model(X_batch)
					loss = self.loss_function(tf.cast(y_batch, tf.float32), y_pred)
					grads = gt.gradient(loss, self.base_model.trainable_variables)
					gradients.append(grads)
					self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))
			
			self.interpreter_callback(gradients[-1])
			

class Interpreter_Custom_Callback:

	def __init__(self, base_model, part_a, algo, neuron_number, feature_embeds, num_neurons):
		self.model = base_model
		self.part_a = part_a
		self.algo = algo
		self.neuron_number = neuron_number
		self.predict_all_neuron = (self.neuron_number == -1)
		self.feature_embeds = feature_embeds
		self.num_neurons = num_neurons
		self.interpreter_output_idxs = []
		self.interpreter_output_words = []
		self.interpreter_output_sims = []
		if self.predict_all_neuron:
			for i in range(self.num_neurons):
				self.interpreter_output_idxs.append([])
				self.interpreter_output_words.append([])
				self.interpreter_output_sims.append([])

	def __call__(self, raw_grads):
		w_and_b = self.model.layers[0].weights
		weights = w_and_b[0].numpy()
		bias = w_and_b[1].numpy()
		weight_grads = raw_grads[0].numpy()
		bias_grads = raw_grads[1].numpy()
		if self.predict_all_neuron:
			for i in range(bias.shape[0]):
				w = weights[:, i:i+1]
				b = bias[i:i+1]
				w_g = weight_grads[:, i:i+1]
				b_g = bias_grads[i:i+1]

				interpreter_inputs = {
					"embeds": self.feature_embeds,
					"weights": w,
					"biases": b,
					"weight_gradients": w_g,
					"bias_gradients": b_g
				}

				interpreter_outputs = self.part_a(interpreter_inputs)

				most_similar_idx = self.algo(self.feature_embeds, tf.squeeze(interpreter_outputs))
			
				self.interpreter_output_idxs[i].append(most_similar_idx)

				self.interpreter_output_sims[i].append(interpreter_outputs)

				self.interpreter_output_words[i].append(self.algo.vocab[most_similar_idx])
		
		else:

			w = weights[:, self.neuron_number:self.neuron_number+1]
			b = bias[self.neuron_number:self.neuron_number+1]

			interpreter_inputs = {
				"embeds": self.feature_embeds,
				"weights": w,
				"biases": b
			}

			interpreter_outputs = self.part_a(interpreter_inputs)

			most_similar_idx = self.algo(self.feature_embeds, tf.squeeze(interpreter_outputs))

			self.interpreter_output_idxs.append(most_similar_idx)

			self.interpreter_output_sims.append(interpreter_outputs)

			self.interpreter_output_words.append(self.algo.vocab[most_similar_idx])


class Interpreter_Callback_2(tf.keras.callbacks.Callback):
	
	def __init__(self, part_a, algo, neuron_number, feature_embeds, num_neurons):
		super().__init__()
		self.part_a = part_a
		self.algo = algo
		self.neuron_number = neuron_number
		self.predict_all_neuron = (self.neuron_number == -1)
		self.feature_embeds = feature_embeds
		self.num_neurons = num_neurons
		self.interpreter_output_idxs = []
		self.interpreter_output_words = []
		self.interpreter_output_sims = []
		if self.predict_all_neuron:
			for i in range(self.num_neurons):
				self.interpreter_output_idxs.append([])
				self.interpreter_output_words.append([])
				self.interpreter_output_sims.append([])

	def on_train_batch_end(self, batch, logs = None):

		w_and_b = self.model.layers[0].weights
		weights = w_and_b[0].numpy()
		bias = w_and_b[1].numpy()
		if self.predict_all_neuron:
			for i in range(bias.shape[0]):
				w = weights[:, i:i+1]
				b = bias[i:i+1]

				interpreter_inputs = {
					"embeds": self.feature_embeds,
					"weights": w,
					"biases": b
				}

				interpreter_outputs = self.part_a(interpreter_inputs)

				most_similar_idx = self.algo(self.feature_embeds, tf.squeeze(interpreter_outputs))
			
				self.interpreter_output_idxs[i].append(most_similar_idx)

				self.interpreter_output_sims[i].append(interpreter_outputs)

				self.interpreter_output_words[i].append(self.algo.vocab[most_similar_idx])
		
		else:

			w = weights[:, self.neuron_number:self.neuron_number+1]
			b = bias[self.neuron_number:self.neuron_number+1]

			interpreter_inputs = {
				"embeds": self.feature_embeds,
				"weights": w,
				"biases": b
			}

			interpreter_outputs = self.part_a(interpreter_inputs)

			most_similar_idx = self.algo(self.feature_embeds, tf.squeeze(interpreter_outputs))

			self.interpreter_output_idxs.append(most_similar_idx)

			self.interpreter_output_sims.append(interpreter_outputs)

			self.interpreter_output_words.append(self.algo.vocab[most_similar_idx])



	


				


# class Interpreter_Callback(tf.keras.callbacks.Callback):
	
# 	def __init__(self, part_a, algo, neuron_number, feature_embeds):
# 		super().__init__()
# 		self.part_a = part_a
# 		self.algo = algo
# 		self.hidden_states = [None]
# 		self.neuron_number = neuron_number
# 		self.feature_embeds = feature_embeds
# 		self.interpreter_output_idxs = []
# 		self.interpreter_output_words = []

# 	def on_train_batch_end(self, batch, logs = None):
# 		w = self.model.layers[0].weights[0].numpy()[:, self.neuron_number:self.neuron_number+1]
# 		b = self.model.layers[0].weights[1].numpy()[self.neuron_number:self.neuron_number+1]

# 		interpreter_inputs = {
# 			"embeds": self.feature_embeds,
# 			"weights": w,
# 			"biases": b
# 		}

# 		interpreter_outputs, hs = self.part_a(interpreter_inputs, self.hidden_states[-1])

# 		most_similar_idx = self.algo(self.feature_embeds, tf.squeeze(interpreter_outputs))

# 		self.hidden_states.append(hs)

# 		self.interpreter_output_idxs.append(most_similar_idx)

# 		self.interpreter_output_words.append(self.algo.vocab[most_similar_idx])
