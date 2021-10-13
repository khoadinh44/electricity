import tflearn
import tensorflow as tf
from utils.optimizers import rmsprop, adam, adagrad, momentum

def leaky_relu(z, name=None):
  return tf.maximum(0.01 * z, z, name=name)

def network():
  # Build neural network
  net = tflearn.input_data(shape=[None, 4])
  net = tflearn.fully_connected(net, 128)
  net = leaky_relu(net)
  net = tflearn.fully_connected(net, 64)
  net = leaky_relu(net)
  net = tflearn.fully_connected(net, 32)
  net = tflearn.fully_connected(net, 4, activation='softmax')
  net = tflearn.regression(net, optimizer=adam, loss='categorical_crossentropy')
  return net
