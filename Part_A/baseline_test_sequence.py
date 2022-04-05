import json, pandas as pd, numpy as np, tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from helper_functions import *
from base_learner import *
from baseline_model import *



def train_sequence(path_to_params):
	with open(path_to_params, "r") as params_json:
		params = json.load(params_json)
	dataset_generator = load_dataset_generator(params["DATA_PATH"])
	vocab, embed_mat, vocab_to_number = load_embed_and_dictionary(params["VOCAB_PATH"], params["EMBED_PATH"])
	baseline = Baseline_Model()
	interpreter_optimizer = tf.keras.optimizers.Adam()
	meta_train_function(baseline, dataset_generator, vocab, embed_mat, interpreter_optimizer, vocab_to_number, params)

def meta_train_function(meta_interpreter, generator, vocab, embed_mat, interpreter_optimizer, vocab_to_number, params):

	for e in range(params["META_EPOCHS"]):
		print(f"On Epoch {e}")
		for s in tqdm(range(params["META_STEPS"])):
			batch = next(generator)
			feature_embeds = np.array([average_embed(w, embed_mat, vocab_to_number) for w in batch[0].columns]).reshape(1, batch[0].shape[1], -1)
			target_embed = average_embed(batch[2].name, embed_mat, vocab_to_number)[tf.newaxis, ...]
			base_model = get_base_learner(params["BASE_NEURONS"], params["BASE_LAYERS"], params["NUM_CLASSES"])
			base_optimizer = tf.keras.optimizers.Adam()
			
			train_function(base_model, meta_interpreter, feature_embeds, target_embed, base_optimizer, interpreter_optimizer, batch, params)

def train_function(base_learner, meta_interpreter, feature_embeds, target_embed, base_optimizer, interpreter_optimizer, batch, params):
	# TODO: Train base learner and meta interpreter and calculate

	X_train, X_test, y_train, y_test = batch

	train_generator = create_flow(np.float32(X_train), np.float32(y_train), params["BATCH_SIZE"])

	for ep in range(params["LEARNER_EPOCHS"]):
		for st in range(params["LEARNER_STEPS"]):
			train_batch = next(train_generator)
			train_step(base_learner, meta_interpreter, feature_embeds, target_embed, base_optimizer, interpreter_optimizer, train_batch, params)

def train_step(base_learner, baseline_interpreter, feature_embeds, target_embed, base_optimizer, interpreter_optimizer, inputs, params):

	with tf.GradientTape() as learner_tape:

				
		y_pred = base_learner(inputs[0])
		learner_mse_loss = tf.square(tf.cast(inputs[0], tf.float32) - y_pred)

		interpreter_inputs = {
			"embeds": feature_embeds,
			"weights": base_learner.layers[0].weights[0],
			"biases": base_learner.layers[0].weights[1]
		}

		interpreter_true_values = cosine_similarity(feature_embeds, target_embed)
		
		interpreter_outputs = baseline_interpreter(interpreter_inputs)

		interpreter_mse_loss = tf.reduce_sum(tf.square(interpreter_outputs - interpreter_true_values))

		print(interpreter_mse_loss)

		learner_grads = learner_tape.gradient(learner_mse_loss, base_learner.trainable_variables)

		base_optimizer.apply_gradients(zip(learner_grads, base_learner.trainable_variables))

