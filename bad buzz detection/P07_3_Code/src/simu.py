# Data libraries
import pandas as pd
import numpy as np

import tensorflow as tf

# My module
import simu_framework
import models

def simu(run,
        padded_corpus_train,
        padded_corpus_val,
        padded_corpus_test,
        y_train,
        y_val,
        y_test,
        model_params,
        ):

    #################################################
    
    # Retrieve simulation parameters
    n_folds = model_params['model_selection']['n_folds']

    # Instantiate a simulation
    s = simu_framework.Sim()

    # Define the parameters of the simulation
    s.add_parameters(('epochs', 10),
                     ('batch_size', 32),
                     ('learning_rate', 0.01),
                     ('dropout_rate', 0.2),
                     ('l2_reg', 0.0001))

    # Define the configurations to simulate
    s.add_configs({'epochs': model_params['model_selection']['epochs'],
                   'batch_size': model_params['model_selection']['batch_size'],
                   'learning_rate': model_params['model_selection']['learning_rate'],
                   'dropout_rate': model_params['model_selection']['dropout_rate'],
                   'l2_reg': model_params['model_selection']['l2_reg'],
                   },
                 )    
                 
    # Explicit the signature of the simulation function
    s.initialize_model_inputs('epochs', 'batch_size', 'learning_rate', 'dropout_rate', 'l2_reg')

    # Run the simulation
    print("\n****************************************")
    print("Train model")
    print("****************************************")
    model, simu_records = s.run_model(models.design_and_train_model,
                                      padded_corpus_train,
                                      np.array(y_train),
                                      padded_corpus_val,
                                      np.array(y_val),
                                      model_params,
                                      n_folds=n_folds,
                                      run=run,
                                      )

    print("\n****************************************")
    print("Records of the training")
    print("****************************************")
    print(simu_records)

    print('\nSave the training records into model_params')
    model_params['training'] = dict()
    model_params['training']['training_records'] = simu_records

    # Retrain the model under the best configuration 
    print("\n****************************************")
    print("Retrain the model under the best configuration")
    print("****************************************")
    
    print('Load the best configuration')
    best_config = {k:[v] for k,v in simu_records['best_config'].items()}
    s.clean_configs()
    s.add_configs(best_config)

    print('Retrain the model with that configuration')
    model, simu_records = s.run_model(models.design_and_train_model,
                                  padded_corpus_train,
                                  np.array(y_train),
                                  padded_corpus_val,
                                  np.array(y_val),
                                  model_params,
                                  n_folds=1,
                                  run=None,
                                  )
    
    print('\nLoad the model with the best metric over epochs')
    model = tf.keras.models.load_model(model_params['paths']['checkpoint_file'])

    print('Save the trained model in the model file\n')
    model.save(model_params['paths']['model_file'])

    print("\n****************************************")
    print("Compute the metric on the test set")
    print("****************************************")

    print("\nEvaluate the model on the train set (for comparisons)")
    loss, accuracy, auc = model.evaluate(padded_corpus_train, np.array(y_train), verbose=0)
    print(f"Loss = {loss}, Accuracy = {accuracy}, AUC = {auc}")

    run.log('train_set_accuracy', accuracy)

    print("\nEvaluate the model on the test set")
    loss, accuracy, auc = model.evaluate(padded_corpus_test, np.array(y_test), verbose=0)
    print(f"Loss = {loss}, Accuracy = {accuracy}, AUC = {auc}")

    run.log('test_set_accuracy', accuracy)



