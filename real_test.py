import tensorflow as tf
import tflearn
from nn.real_nets import network
import numpy as np
import keras
from utils.data_real_loader import validation_examples, validation_targets

# print('\nEnter WSpeed: ')
# x=float(input())
# inputs=np.array([[1, x]])

model = keras.models.load_model("/content/electricity/save/model.h5")
y_pred = model.predict(validation_examples)
print(f'\nThe prediction power is {y_pred[:10]}\n')
print(validation_targets[:10])
# model = keras.models.load_model("/content/electricity/save/model.h5")
# y_pred = model.predict([validation_examples[:, :2], validation_examples[:, 2:]])
# print(f'\nThe prediction power is {y_pred[:10]}\n')
# print(validation_targets[:10])

