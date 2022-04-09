import json, tensorflow as tf, pandas as pd, numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from helper_functions import *
from base_learner import *
from Approach_1 import *



def train_sequence(path_to_params):
	with open(path_to_params, "r") as params_json:
		params = json.load(params_json)
	dataset_generator = load_dataset_generator(params["DATA_PATH"])
	vocab, embed_mat, vocab_to_number = load_embed_and_dictionary(params["VOCAB_PATH"], params["EMBED_PATH"])
	part_a = Part_A(params["HEADS"], params["QUERY_SIZE"], params["FEATURE_SIZE"], params["BATCH_SIZE"], params["D_MODEL"])
	interpreter_optimizer = tf.keras.optimizers.Adam()
	meta_train_function(part_a, dataset_generator, vocab, embed_mat, interpreter_optimizer, vocab_to_number, params)

	norm_matrix =  get_norm_matrix(embed_mat)

	cos_sim_algo = Cosine_Similarity_Algorithmic_Search(vocab, norm_matrix)

	return part_a, cos_sim_algo

def meta_train_function(meta_interpreter_part_a, generator, vocab, embed_mat, interpreter_optimizer, vocab_to_number, params):

	for e in range(params["META_EPOCHS"]):
		print(f"On Epoch {e}")
		epoch_maes = []
		for s in tqdm(range(params["META_STEPS"])):
			batch = next(generator)
			feature_embeds = np.array([average_embed(w, embed_mat, vocab_to_number) for w in batch[0].columns]).reshape(1, batch[0].shape[1], -1)
			target_embed = average_embed(batch[2].name, embed_mat, vocab_to_number)[tf.newaxis, ...]
			base_model = get_base_learner(params["BASE_NEURONS"], params["BASE_LAYERS"], params["NUM_CLASSES"])
			base_optimizer = tf.keras.optimizers.Adam()
			
			mae_metric = train_function(base_model, meta_interpreter_part_a, feature_embeds, target_embed, base_optimizer, interpreter_optimizer, batch, params)

			epoch_maes.append(mae_metric)
		
		print(f"Epoch Average MAE: {float(sum(epoch_maes)) / len(epoch_maes)}")

def train_function(base_learner, meta_interpreter, feature_embeds, target_embed, base_optimizer, interpreter_optimizer, batch, params):
	# TODO: Train base learner and meta interpreter and calculate

	X_train, X_test, y_train, y_test = batch

	hidden_states = [None]

	train_generator = create_flow(np.float32(X_train), np.float32(y_train), params["BATCH_SIZE"])

	mae_metrics = []

	for ep in range(params["LEARNER_EPOCHS"]):
		for st in range(params["LEARNER_STEPS"]):
			train_batch = next(train_generator)
			hs, mae = train_step(base_learner, meta_interpreter, feature_embeds, target_embed, base_optimizer, interpreter_optimizer, train_batch, params, prev_state = hidden_states[-1])
			hidden_states.append(hs)
			mae_metrics.append(mae)
	
	return float(sum(mae_metrics)) / len(mae_metrics)



def train_step(base_learner, meta_interpreter_part_a, feature_embeds, target_embed, base_optimizer, interpreter_optimizer, inputs, params, prev_state = None):
	# TODO: Train base learner and meta interpreter and calculate 

	with tf.GradientTape() as learner_tape, tf.GradientTape() as meta_tape:

				
		y_pred = base_learner(inputs[0])
		learner_mse_loss = tf.square(tf.cast(inputs[0], tf.float32) - y_pred)

		interpreter_inputs = {
			"embeds": feature_embeds,
			"weights": base_learner.layers[0].weights[0],
			"biases": base_learner.layers[0].weights[1]
		}

		interpreter_true_values = cosine_similarity(feature_embeds, target_embed)
	
		interpreter_outputs, hidden_state = meta_interpreter_part_a(interpreter_inputs, prev_state)

		interpreter_mse_loss = tf.reduce_sum(tf.square(interpreter_outputs - interpreter_true_values))

		interpreter_mae_loss = tf.reduce_mean(tf.abs(interpreter_outputs - interpreter_true_values))

		learner_grads = learner_tape.gradient(learner_mse_loss, base_learner.trainable_variables)

		base_optimizer.apply_gradients(zip(learner_grads, base_learner.trainable_variables))

		interpreter_grads = meta_tape.gradient(interpreter_mse_loss, meta_interpreter_part_a.trainable_variables)

		interpreter_optimizer.apply_gradients(zip(interpreter_grads, meta_interpreter_part_a.trainable_variables))

	return hidden_state, interpreter_mae_loss




# IDEA FOR APPROACH 2:
# USE GRADIENT IN META LSTM CELL