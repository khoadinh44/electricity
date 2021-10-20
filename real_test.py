import tensorflow as tf
from nn.real_nets import network
import numpy as np
import keras
from utils.data_real_loader import validation_examples, validation_targets, min_targets, max_targets
from keras.models import load_model

model = network()
model.summary()
model.load_weights('/content/drive/Shareddrives/newpro112233/electricity/weights/model_5_tuabin_next_0_1.h5') 

y_pred = (np.array(model.predict(validation_examples))*(max_targets+min_targets)) - min_targets

print(f'\nThe prediction powers are \n{y_pred[:10]}\n')
print('\nThe true powers are:\n')
print(validation_targets[:10]*(max_targets+min_targets) - min_targets)
