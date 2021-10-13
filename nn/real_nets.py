import tflearn
import tensorflow as tf
from utils.optimizers import rmsprop, adam, adagrad, momentum
def leaky_relu(z, name=None):
  return tf.maximum(0.01 * z, z, name=name)

loss_obj = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

def network():
  # Build neural network
  net = tflearn.input_data(shape=[None, 4]) 
  net = tflearn.fully_connected(net, 32)
  net = tf.keras.layers.PReLU()(net)
  net = tflearn.fully_connected(net, 16)
  net = tflearn.fully_connected(net, 16)
  net = tflearn.fully_connected(net, 4)
  # net = tf.keras.layers.PReLU()(net)
  net = tflearn.regression(net, optimizer=adam, loss='categorical_crossentropy')
  return net
