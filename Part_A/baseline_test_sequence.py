from baseline_model import Baseline_Model
from base_learner import *
from helper_functions import *
from tqdm import tqdm
import json
import tensorflow as tf
import numpy as np



def train_sequence(path_to_params):
	with open(path_to_params, "r") as params_json:
		params = json.load(params_json)
	dataset_generator = load_dataset_generator(params["DATA_PATH"])
	vocab, embed_mat, vocab_to_number = load_embed_and_dictionary(params["VOCAB_PATH"], params["EMBED_PATH"])
	baseline = Baseline_Model()
	interpreter_optimizer = tf.keras.optimizers.Adam()
	meta_train_function(baseline, dataset_generator, vocab, embed_mat, interpreter_optimizer, vocab_to_number, params)

@tf.function()
def meta_train_function(meta_interpreter, generator, vocab, embed_mat, interpreter_optimizer, vocab_to_number, params):

	for e in range(params["META_EPOCHS"]):
		print(f"On Epoch {e}")
		for s in tqdm(range(params["META_STEPS"])):
			batch = next(generator)
			feature_embeds = np.array([average_embed(w, embed_mat, vocab_to_number) for w in batch[0].columns]).reshape(1, batch[0].shape[1], -1)
			target_embed = np.mean(average_embed(w, embed_mat, vocab_to_number) for w in batch[2].name)
			base_model = get_base_learner(params["BASE_NEURONS"], params["BASE_LAYERS"], params["NUM_CLASSES"])
			base_optimizer = tf.keras.optimizers.Adam()
			
			train_function(base_model, meta_interpreter, feature_embeds, target_embed, base_optimizer, interpreter_optimizer, batch, params)


@tf.function()
def train_function(base_learner, meta_interpreter, feature_embeds, target_embed, base_optimizer, interpreter_optimizer, batch, params):
	# TODO: Train base learner and meta interpreter and calculate

	X_train, X_test, y_train, y_test = batch

	train_generator = create_flow(X_train, y_train, params["BATCH_SIZE"])

	for ep in range(params["LEARNER_EPOCHS"]):
		for st in range(params["LEARNER_STEPS"]):
			train_batch = next(train_generator)
			train_step(base_learner, meta_interpreter, feature_embeds, target_embed, base_optimizer, interpreter_optimizer, train_batch, params)


@tf.function()
def train_step(base_learner, baseline_interpreter, feature_embeds, target_embed, base_optimizer, interpreter_optimizer, inputs, params):

	with tf.GradientTape() as learner_tape:

				
		y_pred = base_learner(inputs[0])
		learner_mse_loss = tf.square(inputs[0] - y_pred)

		interpreter_inputs = {
			"embeds": feature_embeds,
			"weights": base_learner.layers[0].weights[0],
			"biases": base_learner.layers[0].weights[1]
		}

		interpreter_true_values = np.array([cosine_similarity(feature_embeds[f, ...], target_embed) for f in range(feature_embeds.shape[0])])

		interpreter_outputs = baseline_interpreter(interpreter_inputs)

		interpreter_mse_loss = tf.square(interpreter_outputs - interpreter_true_values)

		print(interpreter_mse_loss)

		learner_grads = learner_tape.gradient(learner_mse_loss, base_learner.trainable_variables)

		base_optimizer.apply_gradients(zip(learner_grads, base_learner.trainable_variables))

