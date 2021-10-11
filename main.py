import tensorflow as tf
import numpy as np
import tflearn

from utils import preprocess_features, preprocess_targets

# active GPU
tf.debugging.set_log_device_placement(True)

# Prepare data------------------------------------------------------------------------------------
wind_farm_dataframe = pd.read_csv('data/fakedata2980.csv', sep=",")
# Randomization of the data
wind_farm_dataframe = wind_farm_dataframe.reindex(np.random.permutation(wind_farm_dataframe.index))

# Separation of the data into training and validation
training_dataframe = wind_farm_dataframe.head(2300)
validation_dataframe = wind_farm_dataframe.tail(680)

# Definition of the training data input variables and targets, calling the preprocess function
training_examples = preprocess_features(training_dataframe)
training_targets = preprocess_targets(training_dataframe)

# Definition of the validation data input variables and targets, calling the preprocess function
validation_examples = preprocess_features(validation_dataframe)
validation_targets = preprocess_targets(validation_dataframe)

# Optimizer-------------------------------------------------------------------------------------
optimizer = tf.keras.optimizers.RMSprop()

# Load model------------------------------------------------------------------------------------
def train(data=None, labels=None, num_epochs=50, batch_size=32, show_metric=True, path_saver='electricity/save/')
  model = tflearn.DNN(net)
  model.fit(data, labels, num_epochs=num_epochs, batch_size=batch_size, show_metric=show_metric)
  model.save(path_saver)

train(data=training_examples, labels=training_targets)
