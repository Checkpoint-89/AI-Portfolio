# train.py
import time
import pickle

# Import i/o librairies
import os
import sys
import argparse
import shutil
import importlib

# Import ml librairies
from tensorflow import keras

# Import Azure librairies
from azureml.core import Run
from azureml.core.model import Model

# Import custom librairies
import config
from generate import CityScapes
from model import get_model
from utils import get_paths
from metrics import LossAndErrorCallback

run = Run.get_context()

if __name__ == "__main__":

    #########################################
    # Parse inputs
    #########################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, dest='input_folder', help='Path of the input folder')
    parser.add_argument('--output', type=str, dest='output_folder', help='Path of the output folder')
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder

    model_name = config.model_name

    #########################################
    # Define the directory and file paths
    #########################################
    # Directory paths
    src_dir = input_folder +  r"/src"
    target_dir = input_folder + r"/target"

    model_dir = os.path.join(output_folder, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_file = os.path.join(model_dir, f"model_{model_name}.p")
    model_params_file = os.path.join(model_dir, f"model_params_{model_name}.p")

    print("\n===== FOLDERS AND FILES PATHS =====")
    print("input folder: ", args.input_folder)
    print("output folder: ", args.output_folder)
    print("Images source folder: ", src_dir)
    print("Images target folder: ", target_dir)
    print("Model folder: ", model_dir)
    print("Model file: ", model_file)
    print("Model parameters file: ", model_params_file)
    print("=========================")

    #########################################
    # Define the parameters of the model / simulation
    #########################################
    # Dictionnary storing the parameters of the model
    model_params = dict()

    # Setup parameters of simulation
    model_params['params'] = dict()
    model_params['params']['run_eagerly'] = config.run_eagerly

    # Setup the inputs parameters
    model_params['inputs'] = dict()
    model_params['inputs']['img_size'] = config.img_size
    model_params['inputs']['model_input_size'] = config.model_input_size
    model_params['inputs']['num_classes'] = config.num_classes
    model_params['outputs'] = dict()
    model_params['outputs']['model_output_size'] = config.model_output_size


    # Setup model_params
    # model_params['train_test_split'] = dict()
    # model_params['preprocessing'] = dict()
    # model_params['embedding'] = dict()

    model_params['model'] = dict()
    model_params['model']['type'] = config.model_type
    model_params['model']['trainable_base'] = config.trainable_base
    
    model_params['model_selection'] = dict()
    model_params['model_selection']['true_one_hot'] = config.true_one_hot
    model_params['model_selection']['augmentation'] = config.augmentation
    model_params['model_selection']['loss'] = config.loss
    model_params['model_selection']['optimizer'] = config.optimizer
    model_params['model_selection']['metrics'] = config.metrics
    # model_params['model_selection']['n_folds'] = config.n_folds
    model_params['model_selection']['epochs'] = config.epochs
    model_params['model_selection']['epochs_fine_tuning'] = config.epochs_fine_tuning
    model_params['model_selection']['batch_size'] = config.batch_size
    model_params['model_selection']['steps_per_epoch'] = config.steps_per_epoch
    model_params['model_selection']['validation_steps'] = config.validation_steps
    model_params['model_selection']['learning_rate'] = config.learning_rate
    model_params['model_selection']['learning_rate_fine_tuning'] = config.learning_rate_fine_tuning
    # model_params['model_selection']['dropout_rate'] = config.dropout_rate
    # model_params['model_selection']['l2_reg'] = config.l2_reg

    model_params['model_selection']['es_min_delta'] = config.early_stopping_min_delta
    model_params['model_selection']['es_patience'] = config.early_stopping_patience
    model_params['model_selection']['lr_reduce_factor'] = config.lr_reduce_factor
    model_params['model_selection']['lr_reduce_min_delta'] = config.lr_reduce_min_delta
    model_params['model_selection']['lr_reduce_patience'] = config.lr_reduce_patience
    model_params['model_selection']['lr_reduce_min_lr'] = config.lr_reduce_min_lr

    model_params['paths'] = dict()
    model_params['paths']['model_dir'] = model_dir
    model_params['paths']['model_file'] = model_file
    model_params['paths']['model_params_file'] = model_params_file


    #########################################
    # List paths to images and targets
    #########################################
    src_paths, target_paths = get_paths(src_dir, target_dir, '*_reduced.png', '*_reduced_cats.png')

    #########################################
    # Instanciate the generator
    #########################################

    train_gen = CityScapes(model_params['model_selection']['batch_size'], 
                           model_params['inputs']['model_input_size'],
                           model_params['outputs']['model_output_size'],
                           src_paths['train'],
                           target_paths['train'],
                           model_params['model_selection']['augmentation'],
                           model_params['model_selection']['true_one_hot'],
                           model_params['model']['type'],
                           )

    val_gen = CityScapes(model_params['model_selection']['batch_size'], 
                         model_params['inputs']['model_input_size'],
                         model_params['outputs']['model_output_size'],
                         src_paths['val'],
                         target_paths['val'],
                         False,
                         model_params['model_selection']['true_one_hot'],
                         model_params['model']['type'],)


    #########################################
    # Get the model
    #########################################

    if model_params['model']['type'] == 'unet_like':
        keras.backend.clear_session()
        model = get_model(model_params['inputs']['model_input_size'], 
                          model_params['inputs']['num_classes'])
    
    if model_params['model']['type'] == 'vgg_unet':
        import unet
        importlib.reload(unet)
        from unet import vgg_unet
        model, base_model = vgg_unet(model_params['inputs']['num_classes'],
                            input_height=model_params['inputs']['model_input_size'][0],
                            input_width=model_params['inputs']['model_input_size'][1],
                            encoder_level=3,
                            trainable_base=model_params['model']['trainable_base'])
        base_model.trainable = model_params['model']['trainable_base']
        fine_tuned_layers = ['block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_pool']

    if model_params['model']['type'] == 'resnet50_pspnet':
        import pspnet
        importlib.reload(pspnet)
        from pspnet import resnet50_pspnet
        model, base_model = resnet50_pspnet(model_params['inputs']['num_classes'],  
                                input_height=model_params['inputs']['model_input_size'][0],
                                input_width=model_params['inputs']['model_input_size'][1],
                                trainable_base=model_params['model']['trainable_base'])
        base_model.trainable = model_params['model']['trainable_base']

    if model_params['model']['type'] == 'resnet50_unet':
        import unet
        importlib.reload(unet)
        from unet import resnet50_unet
        model, base_model = resnet50_unet(model_params['inputs']['num_classes'],  
                              input_height=model_params['inputs']['model_input_size'][0],
                              input_width=model_params['inputs']['model_input_size'][1],
                              encoder_level=3,
                              trainable_base=model_params['model']['trainable_base'])
        base_model.trainable = model_params['model']['trainable_base']

    model.summary()

    #########################################
    # Compile the model for training
    #########################################

    model.compile(optimizer=model_params['model_selection']['optimizer'],
                  metrics=model_params['model_selection']['metrics'],
                  loss=model_params['model_selection']['loss'],
                  run_eagerly=model_params['params']['run_eagerly'])
    
    model.optimizer.learning_rate.assign(model_params['model_selection']['learning_rate'])
    print("Training optimizer: ", model.optimizer.learning_rate)

    #########################################
    # Train the model
    #########################################

    # Callbacks
    callbacks = [keras.callbacks.ModelCheckpoint(model_params['paths']['model_file'],
                                                 save_best_only=True,
                                                 verbose=1,),
                 LossAndErrorCallback(run),
                #  keras.callbacks.EarlyStopping(monitor='val_dice_metric',
                #                                mode='max',
                #                                min_delta=model_params['model_selection']['es_min_delta'],
                #                                patience=model_params['model_selection']['es_patience'],),
                 keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                   mode='min',
                                                   min_delta=model_params['model_selection']['lr_reduce_min_delta'],
                                                   patience=model_params['model_selection']['lr_reduce_patience'],
                                                   factor=model_params['model_selection']['lr_reduce_factor'],
                                                   min_lr=model_params['model_selection']['lr_reduce_min_lr'] )
                ]

    # Train the model, doing validation at the end of each epoch.
    model.fit(train_gen,
              epochs=model_params['model_selection']['epochs'], 
              validation_data=val_gen,
              steps_per_epoch = model_params['model_selection']['steps_per_epoch'],
              validation_steps = model_params['model_selection']['validation_steps'],
              callbacks=callbacks)


    #########################################
    # Compile the model for fine tuning
    #########################################

    if model_params['model']['type'] != 'unet_like': 
        if model_params['model']['type'] == 'vgg_unet':
            base_model.trainable = False
            for layer_name in fine_tuned_layers:
                layer = base_model.get_layer(layer_name)
                layer.trainable = True
        else:
            base_model.trainable = True

        model.compile(optimizer=model_params['model_selection']['optimizer'],
                    metrics=model_params['model_selection']['metrics'],
                    loss=model_params['model_selection']['loss'],
                    run_eagerly=model_params['params']['run_eagerly'])

        model.optimizer.learning_rate.assign(model_params['model_selection']['learning_rate_fine_tuning'])
        print("Fine tuning optimizer: ", model.optimizer.learning_rate)

    #########################################
    # Fine Tune the model
    #########################################

    # Fine tune the model, doing validation at the end of each epoch.
    if model_params['model']['type'] != 'unet_like': 
        model.fit(train_gen,
                epochs=model_params['model_selection']['epochs_fine_tuning'], 
                validation_data=val_gen,
                steps_per_epoch = model_params['model_selection']['steps_per_epoch'],
                validation_steps = model_params['model_selection']['validation_steps'],
                callbacks=callbacks)

    #########################################
    # Save & Register the model
    #########################################

    # Save the model
    # The model itself has been saved by ModelCheckpoint
    # Here we complete the model by saving it parameters
    print("\nDump the model parameters in a pickle file")
    model_params['model_selection']['metrics'] = str(model_params['model_selection']['metrics'])
    model_params['model_selection']['loss'] = str(model_params['model_selection']['loss'])
    with open(model_params_file, 'wb') as f:
            pickle.dump(model_params, f)

    # Register the model
    print("\nRegister the model in the workspace")
    model = Model.register(workspace=run.experiment.workspace,
                          model_name=model_name,
                          model_path=model_params['paths']['model_dir'],  # Local file to upload and register as a model.
                          description= 'Segmentation',
                          tags={'area': 'image', 'type': 'segmentation'})