# train.py
import time
import pickle

# Import librairies
import os
import config
import load
import preprocess
import embeddings
import models
import simu
import simu_framework

import os
import argparse

from azureml.core import Run
from azureml.core.model import Model

run = Run.get_context()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-folder', type=str, dest='data_folder', help='Path of the data folder')
    parser.add_argument('--model-name', type=str, dest='model_name', help='Model name')
    args = parser.parse_args()

    data_folder = args.data_folder
    model_name = args.model_name

    print("\n===== DATA FOLDER =====")
    print("DATA FOLDER: ", args.data_folder)
    print(f"Name of the trained model: {model_name}")
    print("================")

    # Directory paths
    ROOT_DIR = data_folder
    DATA_DIR = ROOT_DIR +  r"/data"
    SRC_DIR = ROOT_DIR + r"/src"
    DATA_FILE = ROOT_DIR + r"/data/data.csv"
    EMBEDDING_DIR = ROOT_DIR + r"/embeddings"

    # Files where to store intermediate results
    os.mkdir('./checkpoints')
    CHECKPOINT_FILE = r"./checkpoints/weights.best.hdf5"
    os.mkdir('./models')
    MODEL_DIR = f"./models"
    MODEL_FILE = f"./models/model_{model_name}.p"
    MODEL_PARAMS_FILE = f"./models/model_params_{model_name}.p"

    # Dictionnary storing the parameters of the model
    model_params = dict()

    # Setup model_params
    model_params['model_name'] = config.model_name

    model_params['train_test_split'] = dict()
    model_params['train_test_split']['n_samples'] = config.n_samples
    model_params['train_test_split']['test_size'] = config.test_size
    model_params['train_test_split']['val_test_ratio'] = config.val_test_ratio

    model_params['preprocessing'] = dict()
    model_params['preprocessing']['stemming'] = config.stemming
    model_params['preprocessing']['lemmatization'] = config.lemmatization
    model_params['preprocessing']['pos_tag'] = config.pos_tag
    model_params['preprocessing']['max_df'] = config.max_df
    model_params['preprocessing']['min_df'] = config.min_df

    model_params['embedding'] = dict()
    model_params['embedding']['type'] = config.embedding_type
    model_params['embedding']['output_dim'] = config.embedding_output_dim
    model_params['embedding']['trainable'] = config.embedding_trainable

    model_params['model'] = dict()
    model_params['model']['type'] = config.model_type

    model_params['model_selection'] = dict()
    model_params['model_selection']['n_folds'] = config.n_folds

    model_params['model_selection']['epochs'] = config.epochs
    model_params['model_selection']['batch_size'] = config.batch_size

    model_params['model_selection']['learning_rate'] = config.learning_rate

    model_params['model_selection']['dropout_rate'] = config.dropout_rate
    model_params['model_selection']['l2_reg'] = config.l2_reg

    model_params['model_selection']['es_min_delta'] = config.early_stopping_min_delta
    model_params['model_selection']['es_patience'] = config.early_stopping_patience
    model_params['model_selection']['lr_reduce_factor'] = config.lr_reduce_factor
    model_params['model_selection']['lr_reduce_min_delta'] = config.lr_reduce_min_delta
    model_params['model_selection']['lr_reduce_patience'] = config.lr_reduce_patience
    model_params['model_selection']['lr_reduce_min_lr'] = config.lr_reduce_min_lr

    model_params['paths'] = dict()
    model_params['paths']['checkpoint_file'] = CHECKPOINT_FILE
    model_params['paths']['model_dir'] = MODEL_DIR
    model_params['paths']['model_file'] = MODEL_FILE
    model_params['paths']['model_params_file'] = MODEL_PARAMS_FILE
    
    # Load and split data into train and test sets
    corpus_train, corpus_test, y_train, y_test = load.load_and_split_docs(
        DATA_FILE,
        n_tweets=model_params['train_test_split']['n_samples'],
        test_size = model_params['train_test_split']['test_size']
    )

    # Preprocess data
    # preprocessing_inputs = corpus_train, corpus_test, y_train, y_test, val_test_ratio, model_params
    # preprocessing_outputs = preprocess.preprocess(*preprocessing_inputs)
    # vocab_train, tk_doc_max_len, tk_train_corpus, padded_corpus_train, \
    #     padded_corpus_val, padded_corpus_test, y_val, y_test = preprocessing_outputs

    # Preprocess the train set
    preprocessing_inputs= (corpus_train, y_train, model_params)
    tk_train_corpus, padded_corpus_train = preprocess.preprocess_train(*preprocessing_inputs)

    # Preprocess the test set
    preprocessing_inputs = (corpus_test, y_test, model_params)
    padded_corpus_val, padded_corpus_test, y_val, y_test = preprocess.preprocess_test(*preprocessing_inputs)

    # Retrieve variables
    vocab_train = model_params['preprocessing']['vocab_train']
    tk_doc_max_len = model_params['preprocessing']['tk_doc_max_len']

    # Generate embeddings # NOT FUNCTIONAL YET, embeddings are generated outside this script
    if config.compute_dummy_embedding == True:
        embeddings.compute_dummy_embedding(EMBEDDING_DIR, vocab_train)
    if config.load_glove_embedding == True:
        embeddings.load_glove_embedding(DATA_DIR, EMBEDDING_DIR, vocab_train)
    if config.compute_w2v_embedding == True:       
        embeddings.compute_w2v_embedding(EMBEDDING_DIR, tk_train_corpus, vocab_train)
    if config.load_w2v_embedding == True: 
        embeddings.load_w2v_embedding(DATA_DIR, EMBEDDING_DIR, vocab_train)

    # Load embedding
    weights = embeddings.select_embedding(
        EMBEDDING_DIR,
        vocab=vocab_train,
        embedding_type=model_params['embedding']['type']
    )

    # Record model parameter
    model_params['embedding']['weights'] = weights
    model_params['embedding']['output_dim'] = weights.shape[1]

    # Run simu - The trained model is saved in model_params
    simu.simu(
        run,
        padded_corpus_train,
        padded_corpus_val,
        padded_corpus_test,
        y_train,
        y_val,
        y_test,
        model_params,
    )

    # Save the model
    # The model itself has been saved in the simu.py file
    # Here we complete the model by saving it parameters
    print("\nDump the model parameters in a pickle file")
    with open(MODEL_PARAMS_FILE, 'wb') as f:
            pickle.dump(model_params, f)

    # Register the model
    print("\nRegister the model in the workspace")
    model = Model.register(workspace=run.experiment.workspace,
                          model_name=model_params['model_name'],
                          model_path=model_params['paths']['model_dir'],  # Local file to upload and register as a model.
                          description= 'Tweets sentiment analysis',
                          tags={'area': 'tweets', 'type': 'sentiments'})


