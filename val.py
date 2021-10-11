import tensorflow as tf
import tflearn
from nn.nets import network
from utils.data_loader import validation_examples, validation_targets 
from sklearn.metrics import accuracy_score

def validation(weight_path=None, X=None, y_true=None):
  model = tflearn.DNN(network())
  model.load(weight_path)
  y_pred = model.predict(X)
  return accuracy_score(y_true, y_pred)
accuracy = validation('/content/electricity/save/model.tfl', validation_examples, validation_targets)
print(f'Accuracy score of the validation dataset: {accuracy}')
