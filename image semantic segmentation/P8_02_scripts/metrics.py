import tensorflow as tf
from tensorflow import keras
from tensorflow import one_hot
import tensorflow.keras.backend as K
import numpy as np

class LossAndErrorCallback(keras.callbacks.Callback):

    def __init__(self, run):
        super(keras.callbacks.Callback, self).__init__()
        self.run = run

    def on_epoch_end(self, epoch, logs=None):
        for k,v in logs.items():
            self.run.log(k, v)

    # def on_train_batch_end(self, batch, logs=None):
    #     for k,v in logs.items():
    #         self.run.log(k, v)

def dice(weights_):

    def dice_metric(y_true, y_pred):

        y_true = tf.squeeze(y_true, [-1])
        y_true = tf.cast(y_true, dtype='uint8')
        y_true = one_hot(y_true, y_pred.shape[-1])

        epsilon = 0.000001

        y_true = tf.cast(y_true, y_pred.dtype)
        weights = tf.constant(weights_)
        weights = tf.cast(weights, y_pred.dtype)

        num = tf.multiply(y_true, y_pred) * 2.0
        num = tf.reduce_sum(num, [0,1,2])
        den = tf.add(y_true, y_pred)
        den = tf.reduce_sum(den, [0,1,2]) + epsilon
        dices = tf.divide(num,den)

        num = tf.reduce_sum(tf.multiply(weights,dices))
        den = tf.reduce_sum(weights)
        dice_ = tf.divide(num,den)

        return(dice_)
    
    return(dice_metric)


def dice_loss(weights_):

    def custom_loss(y_true, y_pred):

        y_true = tf.squeeze(y_true, [-1])
        y_true = tf.cast(y_true, dtype='uint8')
        y_true = one_hot(y_true, y_pred.shape[-1])

        epsilon = 0.000001

        y_true = tf.cast(y_true, y_pred.dtype)
        weights = tf.constant(weights_)
        weights = tf.cast(weights, y_pred.dtype)

        num = tf.multiply(y_true, y_pred) * 2.0
        num = tf.reduce_sum(num, [0,1,2])
        den = tf.add(y_true, y_pred)
        den = tf.reduce_sum(den, [0,1,2]) + epsilon
        dices = tf.divide(num,den)

        num = tf.reduce_sum(tf.multiply(weights,dices))
        den = tf.reduce_sum(weights)
        dice = tf.divide(num,den)
        
        dice_loss = 1 - dice

        return(dice_loss)
    
    return(custom_loss)


def dice_loss_v2(weights_):

    def custom_loss(y_true, y_pred):

        y_true = tf.squeeze(y_true, [-1])
        y_true = tf.cast(y_true, dtype='uint8')
        y_true = one_hot(y_true, y_pred.shape[-1])

        epsilon = 0.000001

        y_true = tf.cast(y_true, y_pred.dtype)
        weights = tf.constant(weights_)
        weights = tf.cast(weights, y_pred.dtype)

        num = tf.multiply(y_true, y_pred) * 2.0
        num = tf.reduce_sum(num, [0,1,2])
        num = tf.reduce_sum(tf.multiply(weights,num))

        den = tf.add(y_true, y_pred)
        den = tf.reduce_sum(den, [0,1,2])
        den = tf.reduce_sum(tf.multiply(weights,den))+ epsilon
        dice = tf.divide(num,den)
        
        dice_loss = 1 - dice

        return(dice_loss)
    
    return(custom_loss)