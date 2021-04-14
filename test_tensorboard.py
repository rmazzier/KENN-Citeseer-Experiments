import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers

from model import Kenn
from pre_elab import get_train_and_valid_lengths
import settings as s
# Make sure we don't get any GPU errors
# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

writer = tf.summary.create_file_writer("logs/graph_vis")

model = Kenn('knowledge_base')

# def call(self, inputs, **kwargs):
#     features = inputs[0]
#     relations = inputs[1]
#     sx = inputs[2]
#     sy = inputs[3]

#     z = self.preactivations(features)
#     z, _ = self.kenn_layer_1(z, relations, sx, sy)

#     z, _ = self.kenn_layer_2(z, relations, sx, sy)
#     z, _ = self.kenn_layer_3(z, relations, sx, sy)

percentage_of_training = 0.5

##import data
# LOADING DATASET
features = np.load(s.DATASET_FOLDER + 'features.npy')
labels = np.load(s.DATASET_FOLDER + 'labels.npy')

# Import s_x and s_y for the INDUCTIVE learning paradigm
index_x_train = np.load(s.DATASET_FOLDER + 'index_x_inductive_training.npy')
index_y_train = np.load(s.DATASET_FOLDER + 'index_y_inductive_training.npy')
relations_inductive_training = np.load(s.DATASET_FOLDER + 'relations_inductive_training.npy')
index_x_valid = np.load(s.DATASET_FOLDER + 'index_x_inductive_validation.npy')
index_y_valid = np.load(s.DATASET_FOLDER + 'index_y_inductive_validation.npy')
relations_inductive_valid = np.load(s.DATASET_FOLDER + 'relations_inductive_validation.npy')
index_x_test = np.load(s.DATASET_FOLDER + 'index_x_inductive_test.npy')
index_y_test = np.load(s.DATASET_FOLDER + 'index_y_inductive_test.npy')
relations_inductive_test = np.load(s.DATASET_FOLDER + 'relations_inductive_test.npy')

train_len, samples_in_valid = get_train_and_valid_lengths(features, percentage_of_training)



@tf.function
def my_func(inputs):
    training_set = inputs[0]
    relations = inputs[1]
    sx = inputs[2]
    sy = inputs[3]
    return model([training_set, relations, sx, sy])


tf.summary.trace_on(graph=True, profiler=True)
out = my_func([features[:train_len,:], 
    relations_inductive_training, 
    index_x_train, 
    index_y_train])


with writer.as_default():
    tf.summary.trace_export(
        name="function_trace", step=0, profiler_outdir="logs\\graph_vis\\"
    )