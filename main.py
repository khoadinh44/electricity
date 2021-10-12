from __future__ import print_function
import tensorflow as tf
import numpy as np
import tflearn

from utils.data_loader import training_examples, training_targets, validation_examples, validation_targets
from nn.nets import network

# active GPU
tf.debugging.set_log_device_placement(True)

# Load model------------------------------------------------------------------------------------
def train(data=None, labels=None, val_data=None, val_labels=None, network=network, num_epochs=10, batch_size=32, show_metric=True, name_saver=None):
  model = tflearn.DNN(network(), tensorboard_dir='/content/electricity/save/tflearn_logs')
  model.fit(data, labels, n_epoch=num_epochs, batch_size=batch_size, show_metric=show_metric, validation_set=(val_data, val_labels))
  model.save('/content/electricity/save/'+name_saver)

train(training_examples, training_targets, validation_examples, validation_targets, network=network, name_saver='model.tflearn')
