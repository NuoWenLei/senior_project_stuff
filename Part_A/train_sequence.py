import json, tensorflow as tf, pandas as pd, numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from helper_functions import *
from base_learner import *
# from Approach_1 import *
from Approach_5 import *



def train_sequence(path_to_params):
	with open(path_to_params, "r") as params_json:
		params = json.load(params_json)
	dataset_generator = load_dataset_generator(params["DATA_PATH"])
	vocab, embed_mat, vocab_to_number = load_embed_and_dictionary(params["VOCAB_PATH"], params["EMBED_PATH"])
	part_a = Part_A(params["HEADS"], params["QUERY_SIZE"], params["FEATURE_SIZE"], params["BATCH_SIZE"], params["D_MODEL"], params["HIDDEN_SIZE"])
	interpreter_optimizer = tf.keras.optimizers.Adam()
	norm_matrix =  get_norm_matrix(embed_mat)
	cos_sim_algo = Cosine_Similarity_Algorithmic_Search(vocab, norm_matrix)
	logs = meta_train_function(part_a, dataset_generator, vocab, embed_mat, interpreter_optimizer, vocab_to_number, cos_sim_algo, params)

	return part_a, cos_sim_algo, logs

def meta_train_function(meta_interpreter_part_a, generator, vocab, embed_mat, interpreter_optimizer, vocab_to_number, algo, params):

	all_meta_mae = []
	all_learner_mae = []
	all_learner_w_mag = []

	for e in range(params["META_EPOCHS"]):
		print(f"On Epoch {e}")
		epoch_maes = []
		for s in range(params["META_STEPS"]):
			batch = next(generator)
			feature_embeds = np.array([average_embed(w, embed_mat, vocab_to_number) for w in batch[0].columns]).reshape(1, batch[0].shape[1], -1)
			target_embed = average_embed(batch[2].name, embed_mat, vocab_to_number)[tf.newaxis, ...]
			base_model = get_base_learner(params["BASE_NEURONS"], params["BASE_LAYERS"], params["NUM_CLASSES"])
			base_optimizer = tf.keras.optimizers.Adam()
			
			learner_mae, w_mag = train_function(base_model, meta_interpreter_part_a, feature_embeds, target_embed, base_optimizer, interpreter_optimizer, batch, params)

			meta_mae_metric, most_similar_idx = meta_step(base_model, meta_interpreter_part_a, feature_embeds, target_embed, interpreter_optimizer, algo, params)

			print(f"Epoch {e}, Step {s}: {meta_mae_metric:.3f}, Most Similar Word: {algo.vocab[most_similar_idx]}, Target Word: {batch[2].name}")

			epoch_maes.append(meta_mae_metric)

			all_learner_mae.append(learner_mae)
			all_meta_mae.append(meta_mae_metric)
			all_learner_w_mag.append(w_mag)
		
		print(f"Epoch {e} Average MAE: {float(sum(epoch_maes)) / len(epoch_maes)}")
	
	return {
		"learner_mae": all_learner_mae,
		"meta_mae": all_meta_mae,
		"learner_weight_magnitude": all_learner_w_mag
	}

def train_function(base_learner, meta_interpreter, feature_embeds, target_embed, base_optimizer, interpreter_optimizer, batch, params):
	# TODO: Train base learner and meta interpreter and calculate

	X_train, X_test, y_train, y_test = batch

	train_generator = create_flow(np.float32(X_train), np.float32(y_train), params["BATCH_SIZE"])

	test_generator = create_flow(np.float32(X_test), np.float32(y_test), params["BATCH_SIZE"], mode = "test")

	for ep in range(params["LEARNER_EPOCHS"]):
		for st in range(params["LEARNER_STEPS"]):
			train_batch = next(train_generator)
			train_step(base_learner, base_optimizer, train_batch, params)
	avg_learner_weight_mag = learner_weight_magnitude(base_learner, params)
	return test_learner(base_learner, test_generator, params), avg_learner_weight_mag

def meta_step(base_learner, meta_interpreter_part_a, feature_embeds, target_embed, interpreter_optimizer, algo, params):

	with tf.GradientTape() as meta_tape:

		interpreter_inputs = {
			"embeds": feature_embeds,
			"weights": base_learner.layers[0].weights[0],
			"biases": base_learner.layers[0].weights[1]
		}

		interpreter_true_values = cosine_similarity(feature_embeds, target_embed)

		interpreter_outputs = meta_interpreter_part_a(interpreter_inputs)

		most_similar_idx = algo(feature_embeds, tf.squeeze(interpreter_outputs))

		interpreter_mse_loss = tf.reduce_sum(tf.square(interpreter_outputs - interpreter_true_values))

		interpreter_mae_loss = tf.reduce_mean(tf.abs(interpreter_outputs - interpreter_true_values))

		interpreter_grads = meta_tape.gradient(interpreter_mse_loss, meta_interpreter_part_a.trainable_variables)

		interpreter_optimizer.apply_gradients(zip(interpreter_grads, meta_interpreter_part_a.trainable_variables))

	return interpreter_mae_loss, most_similar_idx


def train_step(base_learner, base_optimizer, inputs, params):

	with tf.GradientTape() as learner_tape:

		y_pred = base_learner(inputs[0])

		learner_mse_loss = tf.square(tf.cast(inputs[0], tf.float32) - y_pred)

		learner_grads = learner_tape.gradient(learner_mse_loss, base_learner.trainable_variables)

		base_optimizer.apply_gradients(zip(learner_grads, base_learner.trainable_variables))

def test_learner(base_learner, test_generator, params):
	mae_losses = []
	for X, y in test_generator:
		y_pred = base_learner(X)
		learner_mae_loss = tf.abs(tf.cast(y, tf.float32) - y_pred).numpy()
		mae_losses.append(learner_mae_loss)

	print(f"Learner Test Average MAE: {float(sum(mae_losses)) / float(len(mae_losses)):.3f}")

	return float(sum(mae_losses)) / float(len(mae_losses))

# IDEA FOR APPROACH 2:
# USE GRADIENT IN META LSTM CELL