from train_sequence import tf
from helper_functions import *
from base_learner import *

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
	

class Interpreter_Callback(tf.keras.callbacks.Callback):
	
	def __init__(self, part_a, algo, neuron_number, feature_embeds):
		super().__init__()
		self.part_a = part_a
		self.algo = algo
		self.hidden_states = [None]
		self.neuron_number = neuron_number
		self.feature_embeds = feature_embeds
		self.interpreter_output_idxs = []
		self.interpreter_output_words = []

	def on_train_batch_end(self, batch, logs = None):
		w = self.model.layers[0].weights[0].numpy()[:, self.neuron_number:self.neuron_number+1]
		b = self.model.layers[0].weights[1].numpy()[self.neuron_number:self.neuron_number+1]

		interpreter_inputs = {
			"embeds": self.feature_embeds,
			"weights": w,
			"biases": b
		}

		interpreter_outputs, hs = self.part_a(interpreter_inputs, self.hidden_states[-1])

		most_similar_idx = self.algo(self.feature_embeds, tf.squeeze(interpreter_outputs))

		self.hidden_states.append(hs)

		self.interpreter_output_idxs.append(most_similar_idx)

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
		w_and_b = self.model.layers[0].weights.numpy()
		weights = w_and_b[0]
		bias = w_and_b[1]
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

				self.interpreter_output_sims[i].append(self.algo.vocab[most_similar_idx])
		
		else:

			w = weights[:, i:i+1]
			b = bias[i:i+1]

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



	


				

