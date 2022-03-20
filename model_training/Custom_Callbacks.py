import tensorflow as tf

class CheckpointCallback(tf.keras.callbacks.Callback):

	def __init__(self, save_path, epoch_save_list = [99, 199, 299, 399, 499], starting_epoch = 0):

		super().__init__()

		self.path = save_path
		self.epoch_save_list = epoch_save_list
		self.starting_epoch = starting_epoch

	def on_epoch_end(self, epoch, logs = {}):

		if self.starting_epoch + epoch in self.epoch_save_list:
			self.model.save_weights(f"{self.path}/ep_{self.starting_epoch + epoch + 1}/model")

        