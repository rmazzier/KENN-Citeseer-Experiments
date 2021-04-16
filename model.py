import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.activations import softmax
from parsers import relational_parser
import settings as s


class Standard(Model):
    def __init__(self):
        super(Standard, self).__init__()

    def build(self, input_shape):
        self.h1 = layers.Dense(50, input_shape=input_shape, activation='relu')
        self.d1 = layers.Dropout(0.5)
        self.h2 = layers.Dense(50, input_shape=(50,), activation='relu')
        self.d2 = layers.Dropout(0.5)
        self.h3 = layers.Dense(50, input_shape=(50,), activation='relu')
        self.d3 = layers.Dropout(0.5)

        self.last_layer = layers.Dense(6, input_shape=(50,), activation='linear')

    def preactivations(self, inputs):
        x = self.h1(inputs)
        x = self.d1(x)
        x = self.h2(x)
        x = self.d2(x)
        x = self.h3(x)
        x = self.d3(x)

        return self.last_layer(x)
        # return self.last_layer(inputs)

    def call(self, inputs, **kwargs):
        z = self.preactivations(inputs)

        return z, softmax(z)


class Kenn(Standard):
    """
    Model with 3 KENN layers.
    If self.debug = True, the model returns also the preactivations and the list of all the deltas
    for each individual clause enhancer.
    """
    def __init__(self, knowledge_file, explainer_object=None, *args, **kwargs):
        super(Kenn, self).__init__(*args, **kwargs)
        self.knowledge = knowledge_file
        self.explainer_object = explainer_object

    def build(self, input_shape):
        super(Kenn, self).build(input_shape)
        self.kenn_layer_1 = relational_parser(self.knowledge, explainer_object=self.explainer_object) #returns a RelationalKENN Layer
        self.kenn_layer_2 = relational_parser(self.knowledge, explainer_object=self.explainer_object)
        self.kenn_layer_3 = relational_parser(self.knowledge, explainer_object=self.explainer_object)

    def call(self, inputs, save_debug_data=False, **kwargs):
        features = inputs[0]
        relations = inputs[1]
        sx = inputs[2]
        sy = inputs[3]

        z = self.preactivations(features)
        z, _ = self.kenn_layer_1(z, relations, sx, sy, save_debug_data=save_debug_data)
        z, _ = self.kenn_layer_2(z, relations, sx, sy, save_debug_data=save_debug_data)
        z, _ = self.kenn_layer_3(z, relations, sx, sy, save_debug_data=save_debug_data)

        return softmax(z)


class Kenn_greedy(Model):
    def __init__(self, knowlege_file, debug=False, *args, **kwargs):
        super(Kenn_greedy, self).__init__(*args, **kwargs)
        self.knowledge = knowlege_file
        self.debug = debug
    def build(self, input_shape):
        self.kenn_layer_1 = relational_parser(self.knowledge)

    def call(self, inputs, **kwargs):
        features = inputs[0]
        relations = inputs[1]
        sx = inputs[2]
        sy = inputs[3]

        z, _ = self.kenn_layer_1(features, relations, sx, sy)

        return softmax(z)
