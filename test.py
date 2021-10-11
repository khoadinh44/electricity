import tensorflow as tf
import tflearn
from nn.nets import network
import numpy as np

inputs=np.array([[1, 6.7]])

model = tflearn.DNN(network())
model.load('/content/electricity/save/model.tflearn')
y_pred = model.predict(inputs)
print(f'\nThe prediction is {y_pred[0][1]}')
