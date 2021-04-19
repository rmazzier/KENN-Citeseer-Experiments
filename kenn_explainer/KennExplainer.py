import pickle
import os
import numpy as np
import tensorflow as tf

class KennExplainer(): 
    def __init__(self, debug_data_directory):
        self.debug_data_directory = debug_data_directory
        self.debug_data = {}
        # debug_data will be a dictionary with all the raw data extracted from KENN
        # It will have the following form after we call read_debug_data():
        
        # {
        #   'layer_name_1':{
    #           'unary': {
    #               'deltas':[...],
#                   'preactivations':[...]},
#               'binary':{
#                   'deltas':[...],
#                   'preactivations':[...]},
#               'metadata': {
#                   'unary':[...],
#                   'index1':[...],
#                   'index2':[...],
#                   'unary_predicates':[...],
#                   'unary_clauses':[...],
#                   'binary_predicates':[...],
#                   'binary_clauses':[...]}},

        #     'layer_name_2' etc....
        # }

        # note:
        # len(unary_deltas[i]) = len(unary_predicates) 
        #   i.e. 6 in the citeseer case. Specifically unary_deltas[i] are the
        #   deltas relative to the i-th clause
        # len(binary_deltas[i]) = len(binary_predicates) i.e. 1 in the citeseer case

    def read_debug_data(self):
        for file in os.listdir(self.debug_data_directory):
            data = np.load(self.debug_data_directory+file, allow_pickle=True)

            # layer name is the name of the layer, in case we have more kenn layers
            # i reformat from like this:
            #   'relational_kenn_3'
            # to like this:
            #   'RelationalKenn3'
            layer_name = "".join([word.capitalize() for word in file.split("_")[2:]])[:-4]
            #check if it refers to unary or binary data
            unary_or_binary = file.split("_")[1]
            # whether they are preactivations or deltas or metadata
            data_type = file.split("_")[0]
            self.debug_data.setdefault(layer_name, {})
            if data_type == 'metadata':
                data = list(np.reshape(np.array(data), (1)))[0] #saving a dict with np.save it's not so great...
                self.debug_data[layer_name].setdefault(data_type)
                self.debug_data[layer_name][data_type] = data
                pass
            else:
                self.debug_data[layer_name].setdefault(unary_or_binary, {})
                self.debug_data[layer_name][unary_or_binary].setdefault(data_type)
                self.debug_data[layer_name][unary_or_binary][data_type] = data
    
    def GroupBy(self, deltas, layer_index):
        #to do: actually, i don't need to read unary from the model, i can infer
        # the same shape looking at debug_data['layer1']['unary']['deltas']
        # shape = self.unary.shape
        shape = self.debug_data[list(self.debug_data.keys())[layer_index]]['unary']['deltas'][0].shape
        n_unary = shape[1]
        ux = deltas[:, :n_unary]
        uy = deltas[:, n_unary:2 * n_unary]
        b = deltas[:, 2 * n_unary:]

        index1 = self.debug_data[list(self.debug_data.keys())[layer_index]]['metadata']['index1']
        index2 = self.debug_data[list(self.debug_data.keys())[layer_index]]['metadata']['index2']

        return tf.scatter_nd(index1, ux, shape) + tf.scatter_nd(index2, uy, shape), b

    def get_deltas_from_unary_clause(self, unary_clause_index, layer_index=0):
        """
        Get the deltas generated from a specific unary clause.
        Note that, since they come from unary clauses, the deltas are relative just to unary predicates.
        Parameters:
        - unary_clause_index : The index of the unary clause inside self.unary_clauses;
        - layer_index: The index of the KENN layer of your model, in case you used multiple KENN layers.
                by default it's 0, so there is no need to specify it in case you are using
                just one KENN layer. Input -1 if you want to see the data relative to the final KENN layer.
        """
        key = None
        # First check if we have initialized the data.
        # handle errors if debug_data is still empty
        try:
            key = list(self.debug_data.keys())[0]
        except:
            print("You are trying to get deltas, but you still don't have the data from the training. Have you already trained the model?")
            return

        if layer_index!=0:
            try:
                key = list(self.debug_data.keys())[layer_index]
            except:
                print("You don't have so many KENN layers!")
                print("You have {} KENN layers, but you gave input index {}".format(
                    len(self.debug_data.keys()),
                    layer_index
                ))
                return
        # we have to check the case (like ours with citeseer) in which we have
        # no unary clause...
        if len(self.debug_data[key]['metadata']['unary_clauses'])==0:
            print("You have no unary clauses.")
            return

        # if all is well, just retrieve the deltas from the debug_data.
        # In the unary case, they are ready!
        deltas_from_clause = self.debug_data[key]['unary']['deltas'][unary_clause_index]
        return deltas_from_clause

    def get_deltas_from_binary_clause(self, binary_clause_index, layer_index=0):
        """
        Get the deltas generated from a specific binary clause.
        Note that, since we are looking for deltas from binary clauses, this function
        will return both the deltas for the relative unary and binary predicates.
        Parameters:
        - binary_clause_index : The index of the binary clause inside self.binary_clauses;
        - layer_index: The index of the KENN layer of your model, in case you used multiple KENN layers.
                by default it's 0, so there is no need to specify it in case you are using
                just one KENN layer. Input -1 if you want to see the data relative to the final KENN layer.
        """
        # First check if we have initialized the data.
        # handle errors if debug_data is still empty
        key = None
        try:
            key = list(self.debug_data.keys())[0]
        except:
            print("You are trying to get deltas, but you still don't have the data from the training. Have you already trained the model?")
            return

        if layer_index!=0:
            try:
                key = list(self.debug_data.keys())[layer_index]
            except:
                print("You don't have so many KENN layers!")
                print("You have {} KENN layers, but you gave input index {}".format(
                    len(self.debug_data.keys()),
                    layer_index
                ))
                return

        # we have to check the case in which we have no binary clause:
        if len(self.debug_data[key]['metadata']['binary_clauses'])==0:
            print("You have no binary clauses.")
            return

        # if all is well, we can proceed
        joined_deltas_from_clause = self.debug_data[key]['binary']['deltas'][binary_clause_index]
        grouped_unary_deltas_from_clause, binary_deltas_from_clause = self.GroupBy(joined_deltas_from_clause, layer_index)
        return grouped_unary_deltas_from_clause, binary_deltas_from_clause


    def clear_data(self):
        for file in os.listdir(self.debug_data_directory):
            os.remove(self.debug_data_directory+file)
    
    def _save_data_from_call(self, layer_name, metadata, deltas_u_list, deltas_b_list, u, binary):
        np.save(self.debug_data_directory + 'metadata__'+layer_name, metadata)
        np.save(self.debug_data_directory + 'deltas_unary_' + layer_name, deltas_u_list)
        np.save(self.debug_data_directory + 'deltas_binary_' + layer_name, deltas_b_list)
        np.save(self.debug_data_directory + 'preactivations_unary_' + layer_name, u)
        np.save(self.debug_data_directory + 'preactivations_binary_' + layer_name, binary)
