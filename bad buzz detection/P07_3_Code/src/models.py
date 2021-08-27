# Data libraries
import pandas as pd
import numpy as np

# Neural Network
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import LSTM, GRU, Bidirectional

def design_and_train_model(epochs, batch_size, learning_rate, dropout_rate, l2_reg,
                X_train, y_train,
                X_val, y_val,
                model_params,
                ):

    # Retrieve variables 
    emb_input_length = model_params['preprocessing']['tk_doc_max_len']
    emb_input_dim = len(model_params['preprocessing']['vocab_train'])
    emb_output_dim = model_params['embedding']['output_dim']
    weights = model_params['embedding']['weights']
    type = model_params['model']['type']
    trainable = model_params['embedding']['trainable']
    es_min_delta = model_params['model_selection']['es_min_delta']
    es_patience = model_params['model_selection']['es_patience']
    lr_reduce_factor = model_params['model_selection']['lr_reduce_factor']
    lr_reduce_min_delta = model_params['model_selection']['lr_reduce_min_delta']
    lr_reduce_patience = model_params['model_selection']['lr_reduce_patience']
    lr_reduce_min_lr = model_params['model_selection']['lr_reduce_min_lr']
    checkpoint_file = model_params['paths']['checkpoint_file']

    # Callbacks
    filepath = checkpoint_file
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    earlystopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='auto',
        min_delta=es_min_delta,
        patience=es_patience
    )
    lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=lr_reduce_factor,
        min_delta=lr_reduce_min_delta,
        patience=lr_reduce_patience,
        min_lr=lr_reduce_min_lr
    )
    callbacks = [checkpoint, earlystopping, lr_reduce]

    # Design the model
    if type == 'base':
        embedding_layer = Embedding(input_dim=emb_input_dim,
                        output_dim=emb_output_dim, 
                        input_length=emb_input_length,
                        weights=[weights],
                        trainable=trainable,)
        sequence_input = tf.keras.Input(shape=(emb_input_length,), dtype='int32')
        x = embedding_layer(sequence_input)
        x = Dropout(dropout_rate)(x)
        x = Flatten()(x)
        preds = Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg))(x)
        model = tf.keras.Model(sequence_input, preds)

    if type == 'conv':
        embedding_layer = Embedding(input_dim=emb_input_dim,
                        output_dim=emb_output_dim, 
                        input_length=emb_input_length,
                        weights=[weights],
                        trainable=trainable,)
        sequence_input = tf.keras.Input(shape=(emb_input_length,), dtype='int32')
        x0 = embedding_layer(sequence_input)
        x1 = Conv1D(64, 3, activation='relu', padding='same')(x0)
        x1 = MaxPooling1D(round(emb_input_length))(x1)
        x1 = Flatten()(x1)
        x2 = Conv1D(64, 4, activation='relu', padding='same')(x0)
        x2 = MaxPooling1D(round(emb_input_length))(x2)
        x2 = Flatten()(x2)
        x3 = Conv1D(64, 5, activation='relu', padding='same')(x0)
        x3 = MaxPooling1D(round(emb_input_length))(x3)
        x3 = Flatten()(x3)
        x = tf.keras.layers.Concatenate()([x1, x2, x3])
        x = Dropout(dropout_rate)(x)
        preds = Dense(1, activation='sigmoid',  kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg))(x)
        model = tf.keras.Model(sequence_input, preds)
    
    if type == 'lstm':
        embedding_layer = Embedding(input_dim=emb_input_dim,
                        output_dim=emb_output_dim, 
                        input_length=emb_input_length,
                        weights=[weights],
                        trainable=trainable,)
        sequence_input = tf.keras.Input(shape=(emb_input_length,), dtype='int32')
        x = embedding_layer(sequence_input)
        x = Bidirectional(LSTM(units=64, dropout=0.0, recurrent_dropout=0.0, return_sequences=True))(x)
        x = Bidirectional(LSTM(units=64, dropout=0.0, recurrent_dropout=0.0, return_sequences=False))(x)
        x = Dropout(dropout_rate)(x)
        preds = Dense(1, activation='sigmoid',  kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg))(x)
        model = tf.keras.Model(sequence_input, preds)

    if type == 'lstm+cnn':
        embedding_layer = Embedding(input_dim=emb_input_dim,
                        output_dim=emb_output_dim, 
                        input_length=emb_input_length,
                        weights=[weights],
                        trainable=trainable,)
        sequence_input = tf.keras.Input(shape=(emb_input_length,), dtype='int32')

        x0 = embedding_layer(sequence_input)
        x1 = Conv1D(64, 3, activation='relu', padding='same')(x0)
        x1 = Bidirectional(LSTM(units=64, dropout=0.0, recurrent_dropout=0.0, return_sequences=True))(x1)
        x1 = Bidirectional(LSTM(units=64, dropout=0.0, recurrent_dropout=0.0, return_sequences=False))(x1)
        x1 = Flatten()(x1)

        x2 = Conv1D(64, 4, activation='relu', padding='same')(x0)
        x2 = Bidirectional(LSTM(units=64, dropout=0.0, recurrent_dropout=0.0, return_sequences=True))(x2)
        x2 = Bidirectional(LSTM(units=64, dropout=0.0, recurrent_dropout=0.0, return_sequences=False))(x2)
        x2 = Flatten()(x2)

        x3 = Conv1D(64, 5, activation='relu', padding='same')(x0)
        x3 = Bidirectional(LSTM(units=64, dropout=0.0, recurrent_dropout=0.0, return_sequences=True))(x3)
        x3 = Bidirectional(LSTM(units=64, dropout=0.0, recurrent_dropout=0.0, return_sequences=False))(x3)
        x3 = Flatten()(x3)
        
        x = tf.keras.layers.Concatenate()([x1, x2, x3])
        x = Dropout(dropout_rate)(x)

        preds = Dense(1, activation='sigmoid',  kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg))(x)
    
        model = tf.keras.Model(sequence_input, preds)

    #print(model.summary())

    #Compile the model
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy', 'AUC'])

    # Fit the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks, validation_data=(X_val, y_val), verbose=0)

    return(model)