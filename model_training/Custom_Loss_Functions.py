import tensorflow as tf

def cosine_similarity_loss(x, y):

	# x_embed = tf.nn.embedding_lookup(wine_embed, tf.argmax(x, axis = 1))

	# y_embed = tf.nn.embedding_lookup(wine_embed, tf.argmax(y, axis = 1))

	# tf.argmax(y)

	x_embed = tf.reduce_mean(tf.einsum("ij,jk->ijk", x, wine_embed), axis = 1)

	y_embed = tf.reduce_mean(tf.einsum("ij,jk->ijk", y, wine_embed), axis = 1)
	# Try:
	#   - x as probability
	#   - x as argmaxed


	magnitude_of_x_rows = tf.sqrt(tf.reduce_sum(tf.square(x_embed), axis = -1))
	magnitude_of_y_rows = tf.sqrt(tf.reduce_sum(tf.square(y_embed), axis = -1))

	x_norm = x_embed / magnitude_of_x_rows[:, tf.newaxis]
	y_norm = y_embed / magnitude_of_y_rows[:, tf.newaxis]

	# print(tf.shape(y_norm))
	# print("norm")
	# print(np.isnan(y_norm).sum())
	# print(y_norm)
	# print("mag")
	# print(np.isnan(magnitude_of_y_rows).sum())

	dot_prod = tf.reduce_sum(tf.einsum("ij,ij->ij", x_norm, y_norm), axis = -1)

	# print("dot")
	# print(np.isnan(dot_prod).sum())

	# The lower the loss (the angle), the better the performance

	# print("angle")
	# angle = tf.math.acos(tf.math.abs(dot_prod))
	# print(angle)

	clipped_dot_prod = tf.clip_by_value(dot_prod, clip_value_min=0.001, clip_value_max=0.999)

	cos_sim_loss = -tf.math.log(clipped_dot_prod)

	return tf.reduce_mean(cos_sim_loss)




def magnitude_loss(y_true, y_pred):
	return tf.math.abs(tf.math.sqrt(tf.reduce_sum(tf.square(y_true), axis = -1)) - tf.math.sqrt(tf.reduce_sum(tf.square(y_pred), axis = -1)))




def custom_loss(y_true, y_pred):

	cat_weight = 1e03
	sim_weight = 1e-02
	cat_cross_entropy = tf.keras.losses.CategoricalCrossentropy()

	cat_loss = cat_cross_entropy(y_true, y_pred) * cat_weight

	sim_loss = cosine_similarity_loss(y_true, y_pred) * sim_weight

	return cat_loss + sim_loss