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
    def __init__(self, knowledge_file, *args, **kwargs):
        super(Kenn, self).__init__(*args, **kwargs)
        self.knowledge = knowledge_file

    def build(self, input_shape):
        super(Kenn, self).build(input_shape)
        self.kenn_layer_1 = relational_parser(self.knowledge)
        self.kenn_layer_2 = relational_parser(self.knowledge)
        self.kenn_layer_3 = relational_parser(self.knowledge)

    def call(self, inputs, **kwargs):
        features = inputs[0]
        relations = inputs[1]
        sx = inputs[2]
        sy = inputs[3]

        z = self.preactivations(features)
        z, _ = self.kenn_layer_1(z, relations, sx, sy)
        z, _ = self.kenn_layer_2(z, relations, sx, sy)
        z, _ = self.kenn_layer_3(z, relations, sx, sy)

        return softmax(z)

    def get_weigths(self):
        print()

class Kenn_greedy(Model):
    def __init__(self, knowlege_file, *args, **kwargs):
        super(Kenn_greedy, self).__init__(*args, **kwargs)
        self.knowledge = knowlege_file

    def build(self, input_shape):
        self.kenn_layer_1 = relational_parser(self.knowledge)

    def call(self, inputs, **kwargs):
        features = inputs[0]
        relations = inputs[1]
        sx = inputs[2]
        sy = inputs[3]

        z, _ = self.kenn_layer_1(features, relations, sx, sy)

        return softmax(z)

    def get_weigths(self):
        print()
