import tensorflow as tf
import numpy as np
import pandas as pd
from numpy import genfromtxt
import keras.backend as K
np.random.seed(42)

all_tuabin = np.array([1, 3, 17, 19, 20])
data_path='data/real.XLS'
num_val=50

# Function to define the inputs. Different depending on the model and turbine
def preprocess_features(wind_farm_dataframe):
    selected_features = wind_farm_dataframe[1:, all_tuabin]
    return np.array(selected_features)


def preprocess_targets(wind_farm_dataframe):  
    selected_targets = wind_farm_dataframe[1:, all_tuabin+53]
    return np.array(selected_targets)

# Function used to construct the columns used by the program with the data
def construct_feature_columns(input_features):
    return set([tf.feature_column.numeric_column(my_feature) for my_feature in input_features])

# Function used to load dataset
def input_fn(features, labels, training=True, batch_size=16, num_epochs=1):
        """An input function for training or evaluating"""
        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

        # Shuffle and repeat if you are in training mode.
        if training:
            dataset = dataset.shuffle(1000).repeat()
        dataset = dataset.batch(batch_size).repeat(num_epochs)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

# Prepare data------------------------------------------------------------------------------------
wind_farm_dataframe = np.array(genfromtxt(data_path, delimiter='\t'))
examples = preprocess_features(wind_farm_dataframe)
targets = preprocess_targets(wind_farm_dataframe)
targets = np.where(targets>=0, targets, 0).astype(np.float32)
# max_targets = np.max(targets)
# targets /= max_targets

all_data = np.concatenate((examples, targets), axis=-1).astype(np.float32)
# min_targets = np.abs(np.min(targets))
# max_targets = np.abs(np.max(targets))
# target += min_target
# target /= (max_target+min_target)
# print(f'The min of power is: {min_targets}')
# print(f'The max of power is: {max_targets}\n\n')

val_indices = np.random.choice(all_data.shape[0], size=num_val, replace=False)
train_indices = [i for i in range(len(all_data)) if i not in val_indices]
test_indices = range(308, 318)

all_test = all_data[test_indices]
test_examples = all_test[:, :len(all_tuabin)]
test_targets = all_test[:, len(all_tuabin):]

all_train = all_data[train_indices]
training_examples = all_train[:, :len(all_tuabin)]
training_targets = all_train[:, len(all_tuabin):]

all_validation = all_data[val_indices]
validation_examples = all_validation[:, :len(all_tuabin)]
validation_targets = all_validation[:, len(all_tuabin):]
