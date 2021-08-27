ROOT_DIR = r"D:\Data\Google Drive\Openclassrooms\P7\Projet"
SRC_DIR = ROOT_DIR + r"\src"
DATA_FILE = ROOT_DIR + r"\data\data.csv"

from tensorflow.keras.models import load_model

import sys
sys.path.append(SRC_DIR)

import os
import pickle
import json
import config
import numpy as np

import load
import preprocess

def init():
    global model
    global model_params

    model_name = config.model_name
    # model_params_file_name = os.path.join(os.getcwd(), 'models', f'model_params_{model_name}.p')
    # model_file_name = os.path.join(os.getcwd(), 'models', f'model_params_{model_name}.p')
    model_params_file_name = f'models/model_params_{model_name}.p'
    model_file_name = f'models/model_{model_name}.p'

    path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), model_params_file_name)
    model_params = pickle.load(open(path, "rb"))
    print("Model params: ", model_params)
    path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), model_file_name)
    model = load_model(path)

def run(data):
    try:
        # Set up  the parameters of the model
        model_params['train_test_split']['n_samples'] = 10
        model_params['train_test_split']['test_size'] = 1.0
        model_params['train_test_split']['val_test_ratio'] = 0.0
        print('\nParameters of the model set-up')

        # Load the data
        corpus_test = json.loads(data)['data']
        y_test = np.zeros(len(corpus_test))
        print('\nData loaded')

        # Preprocess the data
        preprocessing_inputs = (corpus_test, y_test, model_params)
        padded_corpus_val, padded_corpus_test, y_val, y_test = preprocess.preprocess_test(*preprocessing_inputs)
        print('\nData preprocessed')

        # Predict the data
        result = model.predict(padded_corpus_test)
        result = ["positive" if r > 0.5 else "negative" for r in result]
        result = [(result[i], corpus_test[i]) for i in range(len(result))]
        
        return(result)
        
    except Exception as e:
        error = str(e)
        return error