import tensorflow as tf

# Function to define the inputs. Different depending on the model and turbine
def preprocess_features(wind_farm_dataframe):
    selected_features = wind_farm_dataframe[["WSpeed_1"]] 
    return selected_features


def preprocess_targets(wind_farm_dataframe):  
# Function to define the target of the neuralnetwork. “Power_1” changes depending on the wind turbine
    selected_targets = wind_farm_dataframe[["Power_1"]]
    return selected_targets

# Function used to construct the columns used by the program with the data
def construct_feature_columns(input_features):
    return set([tf.feature_column.numeric_column(my_feature) for my_feature in input_features])

