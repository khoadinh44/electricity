from __future__ import print_function
import tensorflow as tf
import numpy as np
import tflearn

from utils.data_real_loader import training_examples, training_targets, validation_examples, validation_targets
from nn.real_nets import network
# active GPU
tf.debugging.set_log_device_placement(True)

num_epochs = 200000
batch_size = 32

def train(data=None, labels=None, val_data=None, val_labels=None, network=network, num_epochs=num_epochs, batch_size=batch_size, show_metric=True, name_saver=None):
  model = network()
  model.compile(loss="mean_squared_error",
                metrics=[tf.keras.metrics.Precision()],
                optimizer=tf.keras.optimizers.Adam())

  history = model.fit(data, labels, epochs=num_epochs,
                    validation_data=(val_data, val_labels))
  model.save('/content/drive/Shareddrives/newpro112233/electricity/weights/'+name_saver)

# def train(data=None, labels=None, val_data=None, val_labels=None, network=network, num_epochs=num_epochs, batch_size=batch_size, show_metric=True, name_saver=None):
#   model = network()
#   model.compile(loss=["binary_crossentropy", "binary_crossentropy"],
#                 metrics=[tf.keras.metrics.Precision()],
#                 optimizer=tf.keras.optimizers.Adam(),
#                 loss_weights=[0.5, 0.5])

#   history = model.fit([data[:, :2], data[:, 2:]], [labels, labels], epochs=num_epochs,
#                     validation_data=([val_data[:, :2], val_data[:, 2:]], [val_labels, val_labels]))
#   model.save('/content/electricity/save/'+name_saver)

train(training_examples, training_targets, validation_examples, validation_targets, network=network, name_saver='model_5_tb.h5')
