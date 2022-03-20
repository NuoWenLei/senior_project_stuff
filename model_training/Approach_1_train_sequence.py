import tensorflow as tf, numpy as np, json
from helper_functions import *
from Custom_Callbacks import *
from Data_Generator import *
from Model_Pipelines import *


def main():

	with open("../params.json", "r") as params_json:
		params = json.load(params_json)

	word_dict, wine_words, wine_embed = load_dictionary(params["PATH_TO_VOCAB"], params["PATH_TO_EMBEDDING"])

	rand_features, rand_targets = generate_word_sets_from_tokens_and_embeddings(wine_words, params["DATA_CYCLES"], params["FEATURE_SIZE"], verbose = params["VERBOSE"])

	X_train, X_val, X_test, y_train, y_val, y_test = manual_sample_split(rand_features, rand_targets, len(wine_words), val = True)

	approach_1_pipe = Approach_1_Pipeline(params["QUERY_SIZE"], params["HEADS"], params["D_MODEL"], params["BATCH_SIZE"], params["SEQUENCE_LENGTH"], params["D_FF"], len(wine_words))

	adam = tf.keras.optimizers.Adam(learning_rate=0.01)

	approach_1_pipe.compile(
		loss = "categorical_crossentropy",
		optimizer = adam,
		metrics = ["accuracy"]
	) 

	# Train

	train_generator = Data_Generator(X_train, y_train, wine_embed, len(wine_words), params["BATCH_SIZE"])

	STEPS_PER_EPOCH = len(X_train) // params["BATCH_SIZE"]

	approach_1_pipe.fit(
		x = train_generator,
		batch_size = params["BATCH_SIZE"],
		epochs = params["EPOCHS"],
		steps_per_epoch = STEPS_PER_EPOCH,
		callbacks = [
			CheckpointCallback(params["MODEL_SAVE_PATH"], starting_epoch = params["STARTING_EPOCH"], epoch_save_list = params["EPOCH_SAVE_LIST"])
		]
	)

	# Test

	test_generator = Data_Generator(X_test, y_test, wine_embed, len(wine_words), params["BATCH_SIZE"], mode = "test")

	TEST_STEPS = len(X_test) // params["BATCH_SIZE"]

	approach_1_pipe.evaluate(
		x = test_generator,
		batch_size = params["BATCH_SIZE"],
		steps = TEST_STEPS
	)


if __name__ == "__main__":
	main()



