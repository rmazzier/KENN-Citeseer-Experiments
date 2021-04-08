import tensorflow as tf
import numpy as np
import settings as s

def accuracy(predictions, labels):
    # Accuracy
    correctly_classified = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
    return tf.reduce_mean(tf.cast(correctly_classified, tf.float32))

def Callback_EarlyStopping(AccList, min_delta=s.ES_MIN_DELTA, patience=s.ES_PATIENCE):
    """
    Takes as argument the list with all the validation accuracies. 
    If patience=k, checks if the mean of the last k accuracies is higher than the mean of the 
    previous k accuracies (i.e. we check that we are not overfitting). If not, stops learning.
    """
    #No early stopping for 2*patience epochs 
    if len(AccList)//patience < 2 :
        return False
    #Mean loss for last patience epochs and second-last patience epochs
    mean_previous = np.mean(AccList[::-1][patience:2*patience]) #second-last
    mean_recent = np.mean(AccList[::-1][:patience]) #last
    delta = mean_recent - mean_previous

    if delta <= min_delta:
        print("*CB_ES* Validation Accuracy didn't increase in the last %d epochs"%(patience))
        print("*CB_ES* delta:", delta)
        return True
    else:
        return False