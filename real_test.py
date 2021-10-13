import tensorflow as tf
import tflearn
from nn.real_nets import network
import numpy as np

print('\nEnter WSpeed: ')
inputs = np.array([[ 7.9,  8.9,  8.9,  9.3],
 [10.6, 11.5,  7.8, 10.7],
 [ 3.9,  4.6,  3.9,  3.6],
 [ 2.,   1.9,  1.7,  0.8],
 [11.3, 12.6, 10.2, 11.7],
 [ 2.9,  2.5,  1.5,  1.9],
 [ 5.3,  5.1,  5.4,  5.7],
 [ 9.6,  9.4,  7.8,  8.5],
 [11.6, 12.6, 11.5, 12.8],
 [12.2, 13.4, 12.3, 13.4]])
one_val = np.ones((inputs.shape[0], 196))
validation_examples = np.concatenate((inputs, one_val), axis=1).astype(np.float32)

model = tflearn.DNN(network())
model.load('/content/electricity/save/model.tflearn')
y_pred = model.predict(validation_examples)
print(f'\nThe prediction power is {y_pred}')
