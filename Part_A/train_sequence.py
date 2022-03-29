from base_learner import *
from Approach_1 import *
from helper_functions import *
import json
import tensorflow as tf
import numpy as np

def train_sequence(path_to_params):
	with open(path_to_params, "r") as params_json:
		params = json.load(params_json)
	X_train, X_test, y_train, y_test = load_data(params["data_path"])
	base_model = get_base_learner(params["BASE_NEURONS"], params["BASE_LAYERS"], params["NUM_CLASSES"])
	part_a = Part_A(params["HEADS"], params["QUERY_SIZE"], params["FEATURE_SIZE"], params["BATCH_SIZE"], params["D_MODEL"])

@tf.function()
def train_step():
	# TODO
	pass

# IDEA FOR APPROACH 2:
# USE GRADIENT IN META LSTM CELL