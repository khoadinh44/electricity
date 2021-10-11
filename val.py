import tensorflow as tf
import tflearn
from nets.nn import network
from utils.data_loader import validation_examples, validation_targets 


def validation(weight_path=None, X=None, y=None)
  model = tflearn.DNN(network)
  model.load(weight_path)
  y_pred = model.predict(X)
  return accuracy_score(y_true, y_pred)

print(f'Accuracy score of the validation dataset: {validation(weight_path='model.tflearn', X=validation_examples, y=validation_targets)}')
