import pickle
import os
import numpy as np

class KennExplainer():
    
    def __init__(self, debug_data_directory):
        self.debug_data_directory = debug_data_directory
        # self.knowledge_file = knowledge_file
        self.debug_data = {}

        self.unary_predicates = None
        self.binary_predicates = None
        self.unary_clauses = None
        self.binary_clauses = None

        pass

    def read_debug_data(self):
        for file in os.listdir(self.debug_data_directory):
            # data can be both preactivations data or deltas data
            data = np.load(self.debug_data_directory+file)
            # layer name is the name of the layer, in case we have more kenn layers
            layer_name = file.split("_")[-1]
            #data type means if it refers to unary or binary data
            data_type = file.split("_")[-2]
            self.debug_data.setdefault(layer_name, {})
            self.debug_data[layer_name].setdefault(data_type)
            self.debug_data[layer_name][data_type] = data
    
    def get_predicates_and_clauses(self, unary_predicates,binary_predicates,unary_clauses,binary_clauses):
        self.unary_predicates = unary_predicates
        self.binary_predicates = binary_predicates
        self.unary_clauses = unary_clauses
        self.binary_clauses = binary_clauses
    
    def clear_data(self):
        for file in os.listdir(self.debug_data_directory):
            os.remove(self.debug_data_directory+file)
