import tensorflow as tf
import tflearn
from nn.nets import network
import numpy as np

print('\nEnter WSpeed: ')
x=float(input())
inputs=np.array([[1, x]])

model = tflearn.DNN(network())
model.load('/content/electricity/save/model.tflearn')
y_pred = model.predict(inputs)
print(f'\nThe prediction power is {y_pred[0][1]}')
