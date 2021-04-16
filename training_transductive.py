import tensorflow as tf
from tensorflow import keras
import numpy as np
from model import Standard, Kenn
import os
import settings as s
from training_helper_functions import *
from training_standard import train_and_evaluate_standard

from pre_elab import generate_dataset, get_train_and_valid_lengths

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def train_and_evaluate_kenn_transductive(percentage_of_training, verbose=True):
    """
    Trains KENN model with the Training Set using the Transductive Paradigm.
    Validates on Validation Set, and evaluates accuracy on the Test Set.

    :param debug: if True, the models will return the deltas from the single clause enhancers
    """
    kenn_model = Kenn('knowledge_base', debug=debug)
    kenn_model.build((s.NUMBER_OF_FEATURES,))

    optimizer = keras.optimizers.Adam()
    loss = keras.losses.CategoricalCrossentropy(from_logits=False)

    # LOADING DATASET
    features = np.load(s.DATASET_FOLDER + 'features.npy')
    labels = np.load(s.DATASET_FOLDER + 'labels.npy')

    # Import s_x and s_y for the TRANSDUCTIVE learning paradigm
    index_x = np.load(s.DATASET_FOLDER + 'index_x_transductive.npy')
    index_y = np.load(s.DATASET_FOLDER + 'index_y_transductive.npy')
    relations = np.load(s.DATASET_FOLDER + 'relations_transductive.npy')

    train_len, samples_in_valid = get_train_and_valid_lengths(features, percentage_of_training)

    train_losses = []
    valid_losses = []
    valid_accuracies = []
    train_accuracies = []

    # list of all the evolutions of the clause weights
    clause_weights_1 = []
    clause_weights_2 = []
    clause_weights_3 = []

    train_indices = range(train_len)
    valid_indices = range(train_len, train_len + samples_in_valid)
    test_indices = range(train_len + samples_in_valid, features.shape[0])

    # TRAIN AND EVALUATE KENN MODEL
    for epoch in range(s.EPOCHS_KENN):
        train_step_kenn_transductive(
            model=kenn_model,
            features=features,
            relations=relations,
            index_x_train=index_x,
            index_y_train=index_y,
            labels=labels[train_indices,:],
            loss=loss,
            optimizer=optimizer
        )

        kenn_predictions = kenn_model([features, relations, index_x, index_y])

        train_predictions = kenn_predictions[:train_len,:]
        validation_predictions = kenn_predictions[train_len:(train_len+samples_in_valid),:]

        train_loss = loss(train_predictions, labels[:train_len,:])
        validation_loss = loss(validation_predictions, labels[train_len:(train_len+samples_in_valid),:])

        train_accuracy = accuracy(train_predictions, labels[train_indices, :])
        validation_accuracy = accuracy(validation_predictions, labels[valid_indices, :])

        # Append current clause weights
        c_enhancers_weights_1 = [float(tf.squeeze(ce.clause_weight)) for ce in kenn_model.kenn_layer_1.binary_ke.clause_enhancers]
        clause_weights_1.append(c_enhancers_weights_1)
        c_enhancers_weights_2 = [float(tf.squeeze(ce.clause_weight)) for ce in kenn_model.kenn_layer_2.binary_ke.clause_enhancers]
        clause_weights_2.append(c_enhancers_weights_2)
        c_enhancers_weights_3 = [float(tf.squeeze(ce.clause_weight)) for ce in kenn_model.kenn_layer_3.binary_ke.clause_enhancers]
        clause_weights_3.append(c_enhancers_weights_3)

        # update lists
        train_losses.append(train_loss)
        valid_losses.append(validation_loss)

        train_accuracies.append(train_accuracy)
        valid_accuracies.append(validation_accuracy)
        
        if verbose and epoch%10 == 0:
            print(
                "Epoch {}: Training Loss: {:5.4f} Validation Loss: {:5.4f} | Train Accuracy: {:5.4f} Validation Accuracy: {:5.4f};".format(
                    epoch, train_loss, validation_loss, train_accuracy, validation_accuracy))

        # Early Stopping
        stopEarly = callback_early_stopping(valid_accuracies)
        if stopEarly:
            print("callback_early_stopping signal received at epoch= %d/%d"%(epoch,s.EPOCHS))
            print("Terminating training ")
            break
    # NOTE, IN THE TRANSDUCTIVE CASE THE DELTAS FOR THE BINARY CLAUSES
    # WILL HAVE ALL THE COUPLES (not only the ones in the training set)
    # WE WILL HAVE TO FILTER AFTER MAKING THE GROUP BY
    # select only the deltas relative to test samples
    # for key in ce_deltas.keys():
    #     ce_deltas[key][0] = ce_deltas[key][0][(train_len + samples_in_valid):,:]
        
    kenn_predictions = kenn_model([features, relations, index_x, index_y])
    kenn_test_predictions = kenn_predictions[(train_len + samples_in_valid):,:]
    test_accuracy = accuracy(kenn_test_predictions, labels[(train_len + samples_in_valid):,:])
    all_clause_weights = np.array([clause_weights_1, clause_weights_2, clause_weights_3])

    print("Test Accuracy: {}".format(test_accuracy))
    return {
        "train_losses": train_losses, 
        "train_accuracies": train_accuracies, 
        "valid_losses": valid_losses, 
        "valid_accuracies": valid_accuracies, 
        "test_accuracy": test_accuracy,
        "clause_weights": all_clause_weights,
        "kenn_test_predictions":kenn_test_predictions,
        }

if __name__ == "__main__":
    random_seed=0
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)

    generate_dataset(0.9)
    history_kenn = train_and_evaluate_kenn_transductive(0.9, debug=True)