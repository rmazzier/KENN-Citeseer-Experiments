import tensorflow as tf
from tensorflow import keras
import numpy as np
from model import Standard, Kenn_greedy
import os
import settings as s
from training_functions import accuracy, Callback_EarlyStopping

from pre_elab import generate_dataset, get_train_and_valid_lengths

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def train_step_standard(model, features, labels, loss, optimizer, train_indices):
    with tf.GradientTape() as tape:
        _, predictions = model(features[train_indices,:])
        training_loss = loss(predictions, labels[train_indices,:])

        gradient = tape.gradient(training_loss, model.variables)
        optimizer.apply_gradients(zip(gradient, model.variables))

def train_step_kenn(model, features, relations, index_x_train, index_y_train, labels, loss, optimizer, train_len):
    with tf.GradientTape() as tape:
        predictions_KENN = model([features, relations, index_x_train, index_y_train])
        l = loss(predictions_KENN[:train_len,:], labels[:train_len,:])

        gradient = tape.gradient(l, model.variables)
        optimizer.apply_gradients(zip(gradient, model.variables))

def validation_step_standard(model, features, labels, loss, valid_indices):
    _, predictions = model(features[valid_indices,:])
    valid_loss = loss(predictions, labels[valid_indices,:])
    return predictions, valid_loss

def validation_step_kenn(model, features, relations, index_x_valid, index_y_valid, labels, loss, valid_indices):
    predictions = model([features, relations, index_x_valid, index_y_valid])
    valid_loss = loss(predictions, labels[valid_indices,:])
    return predictions, valid_loss
    return

def _train_and_evaluate_standard_greedy(percentage_of_training, verbose=True):
    """
    THIS FUNCTION IS AUTOMATICALLY CALLED BY train_and_evaluate_kenn_inductive_greedy
    
    Trains Standard model with the Training Set, validates on Validation Set
    and evaluates accuracy on the Test Set.

    NB: The "greedy" keyword just means that we will return the preactivations coming from the whole dataset, 
    after training is finished. Those are needed to train KENN with the greedy approach.
    """
    standard_model = Standard()
    standard_model.build((s.NUMBER_OF_FEATURES,))

    optimizer = keras.optimizers.Adam()
    loss = keras.losses.CategoricalCrossentropy(from_logits=False)

    # LOADING DATASET
    features = np.load(s.DATASET_FOLDER + 'features.npy')
    labels = np.load(s.DATASET_FOLDER + 'labels.npy')

    train_len, samples_in_valid = get_train_and_valid_lengths(features, percentage_of_training)

    train_losses = []
    valid_losses = []
    valid_accuracies = []
    train_accuracies = []

    train_indices = range(train_len)
    valid_indices = range(train_len, train_len + samples_in_valid)
    test_indices = range(train_len + samples_in_valid, features.shape[0])

    # TRAIN AND EVALUATE STANDARD MODEL
    for epoch in range(s.EPOCHS):


        train_step_standard(
            model=standard_model,
            features=features, 
            labels=labels, 
            loss=loss, 
            optimizer=optimizer, 
            train_indices=train_indices)

        _, t_predictions = standard_model(features[train_indices,:])
        t_loss = loss(t_predictions, labels[train_indices,:])
        
        v_predictions, v_loss = validation_step_standard(
            model=standard_model,
            features=features, 
            labels=labels, 
            loss=loss,  
            valid_indices=valid_indices)

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
        stopEarly = Callback_EarlyStopping(valid_accuracies, min_delta=0.001, patience=10)
        if stopEarly:
            print("Callback_EarlyStopping signal received at epoch= %d/%d"%(epoch,s.EPOCHS))
            print("Terminating training ")
            break


    preactivations_nn, predictions_nn = standard_model(features)
    # preactivations_valid, _ = standard_model(features[valid_indices,:])
    predictions_test = predictions_nn[(train_len + samples_in_valid):,:]
    test_accuracy = accuracy(predictions_test, labels[(train_len + samples_in_valid):,:])
    print("Test Accuracy: {}".format(test_accuracy))

    return (
        preactivations_nn, 
        {"train_losses": train_losses, 
        "train_accuracies": train_accuracies, 
        "valid_losses": valid_losses, 
        "valid_accuracies": valid_accuracies, 
        "test_accuracy": test_accuracy})


def train_and_evaluate_kenn_transductive_greedy(percentage_of_training, verbose=True):
    """
    Trains KENN model with the Training Set using the Transductive Paradigm.
    Validates on Validation Set, and evaluates accuracy on the Test Set.
    """
    kenn_model = Kenn_greedy('knowledge_base')
    # kenn_model.build((s.NUMBER_OF_FEATURES,))

    optimizer = keras.optimizers.Adam()
    loss = keras.losses.CategoricalCrossentropy(from_logits=False)

    # LOADING DATASET
    features = np.load(s.DATASET_FOLDER + 'features.npy')
    labels = np.load(s.DATASET_FOLDER + 'labels.npy')

    # TRANSDUCTIVE PARADIGM
    index_x = np.load(s.DATASET_FOLDER + 'index_x_transductive.npy')
    index_y = np.load(s.DATASET_FOLDER + 'index_y_transductive.npy')
    relations = np.load(s.DATASET_FOLDER + 'relations_transductive.npy')


    train_len, samples_in_valid = get_train_and_valid_lengths(features, percentage_of_training)
    
    train_losses = []
    valid_losses = []
    valid_accuracies = []
    train_accuracies = []

    train_indices = range(train_len)
    valid_indices = range(train_len, train_len + samples_in_valid)
    test_indices = range(train_len + samples_in_valid, features.shape[0])

    # before training KENN we need to train the Standard NN
    nn_preactivations, nn_results = _train_and_evaluate_standard_greedy(percentage_of_training)

    ## Now we feed the Kenn_greedy model with the preactivations of the already trained NN, insted of the
    # original features.

    # TRAIN AND EVALUATE KENN MODEL
    for epoch in range(s.EPOCHS_KENN):
        train_step_kenn(
            model=kenn_model,
            features=nn_preactivations,
            relations=relations,
            index_x_train=index_x,
            index_y_train=index_y,
            labels=labels,
            loss=loss,
            optimizer=optimizer,
            train_len = train_len
        )

        kenn_predictions = kenn_model([nn_preactivations, relations, index_x, index_y])

        train_predictions = kenn_predictions[:train_len,:]
        validation_predictions = kenn_predictions[train_len:(train_len+samples_in_valid),:]

        train_loss = loss(train_predictions, labels[:train_len,:])
        validation_loss = loss(validation_predictions, labels[train_len:(train_len+samples_in_valid),:])

        train_accuracy = accuracy(train_predictions, labels[train_indices, :])
        validation_accuracy = accuracy(validation_predictions, labels[valid_indices, :])

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
        stopEarly = Callback_EarlyStopping(valid_accuracies, min_delta=0.001, patience=10)
        if stopEarly:
            print("Callback_EarlyStopping signal received at epoch= %d/%d"%(epoch,s.EPOCHS))
            print("Terminating training ")
            break

    kenn_predictions = kenn_model([nn_preactivations, relations, index_x, index_y])
    test_accuracy = accuracy(kenn_predictions[(train_len + samples_in_valid):,:], labels[(train_len + samples_in_valid):,:])

    print("Test Accuracy: {}".format(test_accuracy))
    kenn_greedy_results= {
        "train_losses": train_losses, 
        "train_accuracies": train_accuracies, 
        "valid_losses": valid_losses, 
        "valid_accuracies": valid_accuracies, 
        "test_accuracy": test_accuracy}
    return (nn_results, kenn_greedy_results)


if __name__ == "__main__":
    generate_dataset(0.75)
    nn_history, greedy_kenn_history = train_and_evaluate_kenn_transductive_greedy(0.75)
