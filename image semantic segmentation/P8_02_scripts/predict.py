import os
import pickle

import json

from tensorflow.keras.models import load_model

from PIL import Image

import config
from metrics import dice_loss, dice
from generate import CityScapes

def init():

    global model
    global model_params
    global image_preprocessor

    model_name = config.model_name
    model_params_file_name = f'models/model_params_{model_name}.p'
    model_file_name = f'models/model_{model_name}.p'

    path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), model_params_file_name)
    with open(path,'rb') as f:
        model_params = pickle.load(f)

    path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), model_file_name)
    model = load_model(path, custom_objects={'custom_loss': dice_loss,'dice_metric': dice,})

    image_preprocessor = CityScapes(img_size=model_params['inputs']['model_input_size'],
                         model_type=model_params['model']['type'],)



import tensorflow as tf
import numpy as np
from labels import labels
from utils import serialize_image, deserialize_image

def run(data):
    try:

        # Read the input of the request
        img = deserialize_image(data)
        size = img.size
        # Preprocess the image
        img = image_preprocessor.preprocess(img)
        img = tf.expand_dims(img, axis=0, name=None)

        # Predict the segmentation mask
        mask = model.predict(img)
        mask = np.argmax(mask, axis=-1).squeeze()

        # Create color code
        cat2color = {label.categoryId: (label.color, label.category) for label in labels}

        response = {'mask':serialize_image(mask), 'mask_size':mask.shape, 'code':cat2color}
        response = json.dumps(response)

    except Exception as e:
        error = str(e)
        return (error)
        
    return(response)
