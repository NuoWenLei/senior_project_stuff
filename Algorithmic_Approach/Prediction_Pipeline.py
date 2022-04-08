from distutils.command.install_egg_info import to_filename
from train_sequence import tf
from helper_functions import *
from base_learner import *

class Interpreted_Learner_Pipeline():

	def __init__(self, part_a, algo, params):
		self.part_a = part_a
		self.algo = algo
		self.base_model = get_base_learner(params["BASE_NEURONS"], params["BASE_LAYERS"], params["NUM_CLASSES"])
	
	def parse(self, neuron_number):
		w = self.base_model.layers[0].weights[0].numpy()[:, neuron_number:neuron_number+1]
		b = self.base_model.layers[0].weights[1].numpy()[neuron_number:neuron_number+1]
		
		pass
	def fit(self, x, y, batch_size, epochs):
		self.base_model.fit(x = x, y = y, batch_size = batch_size, epochs = epochs)