import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

class MultiHeadAttention(Layer):
  def __init__(self, query_size, heads, d_model, batch_size, seq_length, name = "multi_head_attention"):
    super().__init__(name = name)
    self.heads = heads
    self.query_size = query_size
    self.d_model = d_model
    self.batch_size = batch_size
    self.seq_length = seq_length
    self.q_denxse = Dense(self.d_model, activation = "relu", name = f"{name}_dense_q")
    self.k_dense = Dense(self.d_model, activation = "relu", name = f"{name}_dense_k")
    self.v_dense = Dense(self.d_model, activation = "relu", name = f"{name}_dense_v")
    self.final_dense = Dense(self.d_model, activation = "relu", name = f"{name}_dense_final")

    assert d_model == (query_size * heads), f"d_model size incorrect: {d_model} != {query_size} * {heads}"
  
  def split_heads(self, inputs):
    return tf.reshape(inputs, (self.batch_size, self.heads, -1, self.query_size))

  def scaled_dot_product_attention(self, q, k, v):

    distributed_query_key = tf.matmul(q, k, transpose_b = True)

    scaled_distribution = distributed_query_key / tf.math.sqrt(tf.cast(self.query_size, "float32"))

    attention_distribution = tf.nn.softmax(scaled_distribution, axis = -1)

    return tf.matmul(attention_distribution, v)

  def call(self, inputs):
    # Inputs passed in already maps query, key, and value
    # because later multihead attention require q, k, v from different sources
    q, k, v = inputs["query"], inputs["key"], inputs["value"]

    # Linearly transform inputs into Query, Key, and Value
    query = self.q_dense(q)
    key = self.k_dense(k)
    value = self.v_dense(v)

    # Split heads for each of query, key, and value
    split_query = self.split_heads(query)
    split_key = self.split_heads(key)
    split_values = self.split_heads(value)

    # Calculate Scaled Dot Produt Attention
    scaled_attention = self.scaled_dot_product_attention(split_query, split_key, split_values)

    # Concatenate heads of the attention 
    concat_attention = tf.reshape(scaled_attention, (self.batch_size, -1, self.d_model))

    # Do a final linear transformation of the concatenated attention
    outputs = self.final_dense(concat_attention)

    return outputs

class FeedForwardNetwork(Layer):
  def __init__(self, d_ff, d_model, name = "ff"):
    super().__init__(name = name)
    self.d_ff = d_ff
    self.dense_1 = Dense(d_ff, activation = "relu", name = f"{name}_dense_1")

    # Final Layer is the size of d_model
    self.dense_final = Dense(d_model, activation = "relu", name = f"{name}_dense_final")
    self.dropout = Dropout(0.3)
  
  def call(self, inputs):
    x = self.dense_1(inputs)
    x = self.dense_final(x)
    x = self.dropout(x)
    return x

