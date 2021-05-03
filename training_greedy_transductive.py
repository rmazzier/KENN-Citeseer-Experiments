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


def train_and_evaluate_kenn_transductive_greedy(percentage_of_training, verbose=True):
    """
    Trains KENN model with the Training Set using the Transductive Paradigm.
    Validates on Validation Set, and evaluates accuracy on the Test Set.
    """
    kenn_model = Kenn_greedy('knowledge_base')
    kenn_model.build((s.NUMBER_OF_FEATURES,))

    optimizer = keras.optimizers.Adam()
    loss = keras.losses.CategoricalCrossentropy(from_logits=False)

    # LOADING DATASET
    features = np.load(s.DATASET_FOLDER + 'features.npy')
    labels = np.load(s.DATASET_FOLDER + 'labels.npy')

    # TRANSDUCTIVE PARADIGM
    index_x = np.load(s.DATASET_FOLDER + 'index_x_transductive.npy')
    index_y = np.load(s.DATASET_FOLDER + 'index_y_transductive.npy')
    relations = np.load(s.DATASET_FOLDER + 'relations_transductive.npy')

    train_len, samples_in_valid = get_train_and_valid_lengths(
        features, percentage_of_training)

    train_losses = []
    valid_losses = []
    valid_accuracies = []
    train_accuracies = []

    train_indices = range(train_len)
    valid_indices = range(train_len, train_len + samples_in_valid)
    test_indices = range(train_len + samples_in_valid, features.shape[0])

    # before training KENN we need to train the Standard NN
    nn_preactivations_train, nn_preactivations_valid, nn_preactivations_test, nn_results = train_and_evaluate_standard(
        percentage_of_training, verbose=verbose)
    nn_preactivations = np.concatenate(
        (nn_preactivations_train, nn_preactivations_valid, nn_preactivations_test), axis=0)
    # Now we feed the Kenn_greedy model with the preactivations of the already trained NN, insted of the
    # original features.

    # TRAIN AND EVALUATE KENN MODEL
    for epoch in range(s.EPOCHS_KENN):
        train_step_kenn_transductive(
            model=kenn_model,
            features=nn_preactivations,
            relations=relations,
            index_x_train=index_x,
            index_y_train=index_y,
            labels=labels[train_indices, :],
            loss=loss,
            optimizer=optimizer
        )

        kenn_predictions = kenn_model(
            [nn_preactivations, relations, index_x, index_y])

        train_predictions = kenn_predictions[:train_len, :]
        validation_predictions = kenn_predictions[train_len:(
            train_len+samples_in_valid), :]

        train_loss = loss(train_predictions, labels[:train_len, :])
        validation_loss = loss(
            validation_predictions, labels[train_len:(train_len+samples_in_valid), :])

        train_accuracy = accuracy(train_predictions, labels[train_indices, :])
        validation_accuracy = accuracy(
            validation_predictions, labels[valid_indices, :])

        # update lists
        train_losses.append(train_loss)
        valid_losses.append(validation_loss)

        train_accuracies.append(train_accuracy)
        valid_accuracies.append(validation_accuracy)

        if verbose and epoch % 10 == 0:
            print(
                "Epoch {}: Training Loss: {:5.4f} Validation Loss: {:5.4f} | Train Accuracy: {:5.4f} Validation Accuracy: {:5.4f};".format(
                    epoch, train_loss, validation_loss, train_accuracy, validation_accuracy))

        # Early Stopping
        stopEarly = callback_early_stopping(valid_accuracies)
        if stopEarly:
            print("callback_early_stopping signal received at epoch= %d/%d" %
                  (epoch, s.EPOCHS))
            print("Terminating training ")
            break

    kenn_predictions = kenn_model(
        [nn_preactivations, relations, index_x, index_y])
    test_accuracy = accuracy(kenn_predictions[(
        train_len + samples_in_valid):, :], labels[(train_len + samples_in_valid):, :])

    print("Test Accuracy: {}".format(test_accuracy))
    kenn_greedy_results = {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "valid_losses": valid_losses,
        "valid_accuracies": valid_accuracies,
        "test_accuracy": test_accuracy}
    return (nn_results, kenn_greedy_results)


if __name__ == "__main__":
    generate_dataset(0.75)
    nn_history, greedy_kenn_history = train_and_evaluate_kenn_transductive_greedy(
        0.75)
