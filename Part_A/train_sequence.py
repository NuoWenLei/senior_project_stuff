from base_learner import *
from Approach_1 import *
from helper_functions import *
from tqdm import tqdm
import json
import tensorflow as tf
import numpy as np

def train_sequence(path_to_params):
	with open(path_to_params, "r") as params_json:
		params = json.load(params_json)
	dataset_generator = load_dataset_generator(params["data_path"])
	sample_X, _, sample_y, _ = next(dataset_generator)
	vocab, embed_mat, vocab_to_number = load_embed_and_dictionary(params["VOCAB_PATH"], params["EMBED_PATH"])

	part_a = Part_A(params["HEADS"], params["QUERY_SIZE"], params["FEATURE_SIZE"], params["BATCH_SIZE"], params["D_MODEL"])
	meta_train_function(part_a, dataset_generator, vocab, embed_mat, vocab_to_number, params)

@tf.function()
def meta_train_function(meta_interpreter_part_a, generator, vocab, embed_mat, vocab_to_number, params):

	for e in range(params["META_EPOCHS"]):
		print(f"On Epoch {e}")
		for s in tqdm(range(params["META_STEPS"])):
			batch = next(generator)
			feature_embeds = np.array([embed_mat[vocab_to_number[w]] for w in batch[0].columns]).reshape(1, batch[0].shape[1], -1)
			target_embed = np.mean(embed_mat[vocab_to_number[w]] for w in batch[2].name)
			base_model = get_base_learner(params["BASE_NEURONS"], params["BASE_LAYERS"], params["NUM_CLASSES"])
			
			train_function(base_model, meta_interpreter_part_a, feature_embeds, target_embed, batch, params)


@tf.function()
def train_function():
	# TODO: Train base learner and meta interpreter and calculate 

	pass


@tf.function()
def train_step(base_learner, meta_interpreter_part_a, feature_embeds, inputs, params):
	# TODO: Train base learner and meta interpreter and calculate 

	pass

# IDEA FOR APPROACH 2:
# USE GRADIENT IN META LSTM CELL