import tensorflow as tf
from tensorflow import keras
import numpy as np
from model import Standard, Kenn_greedy
import os
import settings as s
from training_helper_functions import *
from training_standard import train_and_evaluate_standard

from pre_elab import generate_dataset, get_train_and_valid_lengths

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def train_and_evaluate_kenn_inductive_greedy(percentage_of_training, verbose=True):
    """
    Trains KENN model with the Training Set using the Inductive Paradigm.
    Validates on Validation Set, and evaluates accuracy on the Test Set.
    """
    kenn_model = Kenn_greedy('knowledge_base')
    kenn_model.build((s.NUMBER_OF_FEATURES,))

    optimizer = keras.optimizers.Adam()
    loss = keras.losses.CategoricalCrossentropy(from_logits=False)

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
    
    train_losses = []
    valid_losses = []
    valid_accuracies = []
    train_accuracies = []

    train_indices = range(train_len)
    valid_indices = range(train_len, train_len + samples_in_valid)
    test_indices = range(train_len + samples_in_valid, features.shape[0])

    # before training KENN we need to train the Standard NN
    nn_train_preactivations, nn_valid_preactivations, nn_test_preactivations, nn_results = train_and_evaluate_standard(percentage_of_training, verbose=verbose)

    ## Now we feed the Kenn_greedy model with the preactivations of the already trained NN, insted of the
    # original features.
    # TRAIN AND EVALUATE KENN MODEL
    for epoch in range(s.EPOCHS_KENN):
        train_step_kenn_inductive(
            model=kenn_model,
            features=nn_train_preactivations,
            relations=relations_inductive_training,
            index_x_train=index_x_train,
            index_y_train=index_y_train,
            labels=labels[train_indices,:],
            loss=loss,
            optimizer=optimizer
        )

        t_predictions = kenn_model([nn_train_preactivations, relations_inductive_training, index_x_train, index_y_train])
        t_loss = loss(t_predictions, labels[train_indices,:])

        v_predictions, v_loss = validation_step_kenn_inductive(
            model=kenn_model,
            features=nn_valid_preactivations,
            relations=relations_inductive_valid,
            index_x_valid=index_x_valid,
            index_y_valid=index_y_valid,
            labels=labels[valid_indices,:],
            loss=loss
        )

        train_losses.append(t_loss)
        valid_losses.append(v_loss)

        t_accuracy=accuracy(t_predictions, labels[train_indices, :])
        v_accuracy=accuracy(v_predictions, labels[valid_indices, :])

        train_accuracies.append(t_accuracy)
        valid_accuracies.append(v_accuracy)
        
        if verbose and epoch%10 == 0:
            print(
                "Epoch {}: Training Loss: {:5.4f} Validation Loss: {:5.4f} | Train Accuracy: {:5.4f} Validation Accuracy: {:5.4f};".format(
                    epoch, t_loss, v_loss, t_accuracy, v_accuracy))

        # Early Stopping
        stopEarly = callback_early_stopping(valid_accuracies)
        if stopEarly:
            print("callback_early_stopping signal received at epoch= %d/%d"%(epoch,s.EPOCHS))
            print("Terminating training ")
            break

    predictions_test = kenn_model([nn_test_preactivations, relations_inductive_test, index_x_test, index_y_test])
    test_accuracy = accuracy(predictions_test, labels[test_indices,:])

    print("Test Accuracy: {}".format(test_accuracy))
    greedy_kenn_results = {
        "train_losses": train_losses, 
        "train_accuracies": train_accuracies, 
        "valid_losses": valid_losses, 
        "valid_accuracies": valid_accuracies, 
        "test_accuracy": test_accuracy}

    return (nn_results, greedy_kenn_results)

if __name__ == "__main__":
    generate_dataset(0.75)
    nn_history, greedy_kenn_history = train_and_evaluate_kenn_inductive_greedy(0.75)
