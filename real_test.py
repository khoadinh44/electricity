import tensorflow as tf
import tflearn
from nn.real_nets import network
import numpy as np

print('\nEnter WSpeed: ')
inputs = np.array([[0.5, 0.7, 1.1, 1.2]])

model = tflearn.DNN(network())
model.load('/content/electricity/save/model.tflearn')
y_pred = model.predict(inputs)
print(f'\nThe prediction power is {y_pred}')
