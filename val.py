import tensorflow as tf
import tflearn
from nn.nets import network
from utils.data_loader import validation_examples, validation_targets 


def validation(weight_path=None, X=None, y=None):
  model = tflearn.DNN(network)
  model.load(weight_path)
  y_pred = model.predict(X)
  return accuracy_score(y_true, y_pred)

accuracy = validation(weight_path='save/model.tflearn', X=validation_examples, y=validation_targets)
print(f'Accuracy score of the validation dataset: {accuracy}')
