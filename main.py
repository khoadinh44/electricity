import tensorflow as tf
import numpy as np
import tflearn

from utils import training_examples, training_targets
from nets.nn import network

# active GPU
tf.debugging.set_log_device_placement(True)

# Load model------------------------------------------------------------------------------------
def train(data=None, labels=None, network=network, num_epochs=50, batch_size=32, show_metric=True, path_saver='electricity/save/')
  model = tflearn.DNN(network)
  model.fit(data, labels, num_epochs=num_epochs, batch_size=batch_size, show_metric=show_metric)
  model.save(path_saver)

train(data=training_examples, labels=training_targets, network=network)
