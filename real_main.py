from __future__ import print_function
import tensorflow as tf
import numpy as np
import tflearn
from utils.data_real_loader import training_examples, training_targets, validation_examples, validation_targets
from nn.real_nets import network

# active GPU--------------------------------------------------------------------------------------------------------------------------
tf.debugging.set_log_device_placement(True)

# parameter-----------------------------------------------------------------------------------------------------------------------------
num_epochs = 4100
batch_size = 32
path_saver = '/content/drive/Shareddrives/newpro112233/electricity/weights/'

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
def train(data=None, labels=None, val_data=None, val_labels=None, network=network, num_epochs=num_epochs, batch_size=batch_size, show_metric=True, name_saver=None):
  model = network()
  model.compile(loss="mean_squared_error",
                metrics=[tf.keras.metrics.Precision()],
                optimizer=tf.keras.optimizers.Adam())

  history = model.fit(data, labels, epochs=num_epochs,
                     validation_data=(val_data, val_labels),
                     callbacks=[callback])
  model.save(path_saver+name_saver)

train(training_examples, training_targets, validation_examples, validation_targets, network, num_epochs, batch_size, False, name_saver='model_5_tuabin.h5')
