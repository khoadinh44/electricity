import tflearn
from utils.optimizers import rmsprop, adam, adagrad, momentum

def network():
  # Build neural network
  net = tflearn.input_data(shape=[None, 4])
  net = tflearn.fully_connected(net, 64)
  net = tflearn.activations.relu(net)
  net = tflearn.fully_connected(net, 32)
  net = tflearn.fully_connected(net, 4, activation='softmax')
  net = tflearn.regression(net, optimizer=adam, loss='categorical_crossentropy')
  return net
