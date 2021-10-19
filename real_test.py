import tensorflow as tf
import tflearn
from nn.real_nets import network
import numpy as np
import keras
from utils.data_real_loader import validation_examples, validation_targets

model = keras.models.load_model("/content/drive/Shareddrives/newpro112233/electricity/weights/model_5_tuabin.h5")
y_pred = model.predict(validation_examples)

print(f'\nThe prediction powers are \n{y_pred[:10]}\n')
print('\nThe true powers are:\n')
print(validation_targets[:10])
