# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 11:27:20 2020

@author: cdiet
"""

import numpy as np
import pandas as pd

class Sim:
    """
    Manages the parameters of the simulation in
    self.parameters = {"param1":default1, "param2":default:2, etc} 
    
    Manages the configs of the simulation in
    self.configs = [{"param1":val1, "param2":val2, etc}, etc]

    """
    
    def __init__(self):
        self.labels = {} #I believe this is obsolete, to be confirmed
        self.parameters = {} #dictionnary parameter: default value
        self.configs = [] #list of dictionnaries, each being a sim config
        self.model_inputs = []
        self.results = []
        
    def add_parameters(self, *args, overwrite = False):
        """
        Adds new parameters and their default values to self.parameters.
        Update configs accordingly.
        
        If a parameter already exists and overwrite == False, do nothing;
        otherwise: overwrites the default value in parameters and leaves 
        configs untouched
        
        Packages: none.
            
        Arguments:
        args -- tupples like (parameter, default value).
        overwrite -- allow overwriting if True as explained above.
        
        Returns:
        Updated self.parameters and self.configs

        """
        for arg in args:
            try: assert type(arg) == tuple
            except AssertionError: print("tupple expected")
            
            try: assert len(arg) == 2
            except AssertionError: print("2 items expected")
            
            try: assert type(arg[0]) == str
            except AssertionError: print("string expected")
                
            parameter = arg[0]
            default = arg[1]
            
            if parameter in self.parameters.keys() and overwrite == False:
                pass
            elif parameter in self.parameters.keys() and overwrite == True:
                self.parameters[parameter] = default
            else:
                self.parameters[parameter] = default
                for i in range(len(self.configs)):
                    self.configs[i][parameter] = default
                    
    def add_config(self,**in_config):
        """
        Adds a new config to the configurations list self.configs
        
        If a key of in_config is not in self.parameters, it is ignored
        If a parameter of self.parameters is not in in_config, it is added to 
        its default value
        
        Packages: none.
            
        Arguments:
        in_config -- sequence of keys and values
        
        Returns:
        Updated self.configs

        """
        #Initializes the new config to default.
        new_config = {key:value for (key,value) in self.parameters.items()}
        #Set-up the new config
        for key in (new_config.keys() & in_config.keys()):
            new_config[key] = in_config[key]
        #Append the new config
        self.configs.append(new_config)
        
    def clean_configs(self):
        """
        Delete all configs.
        
        Packages: none.
            
        Arguments: none
        
        Returns:
        Updated self.configs = []

        """        
        self.configs = []

    def append_dict2list(self, l_in, d_in):
        """
        Append d_in to l_in reflecting the scheme below
        
        With inputs
        l_in=[{"p1":1, "p2":4},{"p1":1, "p2":5},\
              {"p1":2, "p2":4},{"p1":2, "p2":5}] 
        
        d_in={"p3":[6,7]} 

        The output will be:
        
        [{'p1': 1, 'p2': 4, 'p3': 6},
         {'p1': 1, 'p2': 4, 'p3': 7},
         {'p1': 1, 'p2': 5, 'p3': 6},
         {'p1': 1, 'p2': 5, 'p3': 7},
         {'p1': 2, 'p2': 4, 'p3': 6},
         {'p1': 2, 'p2': 4, 'p3': 7},
         {'p1': 2, 'p2': 5, 'p3': 6},
         {'p1': 2, 'p2': 5, 'p3': 7}]
        
        Packages: none.
            
        Arguments:
        l_in - a list of dictionnaries.
        d_in - a dictionary containing only 1 key and 1 value as a list
        
        Returns:
        A list of dictionnaries

        """
        len_l = len(l_in)
        key = list(d_in.keys())[0]
        len_d = len(d_in[key])
        
        l_out = [list(l_in[i].items()) \
                 for i in range(len_l)]
        
        l_out = [l_out[i] + [(key, d_in[key][j])] \
                 for i in range(len_l) for j in range(len_d)]
        
        l_out = [{k: v for (k,v) in l_out[i]} for i in range(len(l_out))]
                
        return(l_out)
    
    def dict2list(self, d_in):
        """
        Flushes a dictionnary of lists into a list of dictionaries.
        After the execution of the function, d_in = {}.
        
        Example: 
            
            d_in = {"p1":[1,2], "p2":[4,5,6]}
            
            results in:
                
            d_in = {}
            output = 
            [{'p1': 1, 'p2': 4},
             {'p1': 2, 'p2': 4},
             {'p1': 1, 'p2': 5},
             {'p1': 2, 'p2': 5},
             {'p1': 1, 'p2': 6},
             {'p1': 2, 'p2': 6}]
        
        Packages: none.
            
        Arguments:
        d_in -- a dictionnary of lists.
        
        Returns:
        A list of dictionnaries.

        """
        if len(d_in) == 0:
            return([{}])
        
        (k,v)=d_in.popitem()
        d1={k:v}
        
        return(self.append_dict2list(self.dict2list(d_in),d1))

    def add_configs(self, d_in):
        """
        Easy way to add configs resulting from the combinations parameters 
        provided as in the example hereunder:
        s.add_configs({"num_iterations":[500, 1500],\
               "learning_rate": [0.005, 0.007]}
        """
        l_configs = self.dict2list(d_in)
        for config in l_configs:
            self.add_config(**config)
        
    def initialize_model_inputs(self, *args):
        """
        Defines what are the input parameters for the function model()
        as strings and in what order they have to be entered.
        Those strings will be used as keys to set-up each simulation
        according to the parameters defined in each config of self.configs.
        """
        
        self.model_inputs = args
        return(None)

        
    def run_model(self, model, X_train, y_train, X_val, y_val, model_params, n_folds, run=None):

        # Setup
        l = len(X_train)
        fold_len = l / n_folds

        records = dict(
            configs = dict(),
            mean_train_losses_history = dict(),
            mean_train_accuracies_history = dict(),
            mean_val_losses_history = dict(),
            mean_val_accuracies_history = dict(),
            best_mean_val_loss = 1,
            best_mean_train_accuracy = 0,
            best_mean_val_accuracy = 0,
            best_config = None,
            best_config_n = None,
            best_epoch_n = None
        )

        # Loop through the configurations
        for i in range(len(self.configs)):

            # Init
            train_loss_history = dict()
            train_accuracy_history = dict()
            val_loss_history = dict()
            val_accuracy_history = dict()
            n_epochs = []

            # Loop through the folds
            for j in range(n_folds):
                print(f"\nConfiguration {i+1} / {len(self.configs)}: {self.configs[i]}")
                print(f"Fold {j+1} / {n_folds}")

                # Compute the index of the validation set for this fold
                idx = np.isin(np.arange(l), np.arange(round(j * fold_len), round((j+1) * fold_len)))

                # Extract the train and validation sets
                Xfold_val = X_train[idx]
                yfold_val = y_train[idx]
                Xfold_train = X_train[~idx]
                yfold_train = y_train[~idx]

                # Extract the train and validation sets in case there is only 1 fold
                if n_folds == 1:
                    Xfold_train = X_train
                    yfold_train = y_train
                    Xfold_val = X_val
                    yfold_val = y_val

                # Format the input of the model
                inputs = [self.configs[i][key] for key in self.model_inputs]
                inputs.extend([Xfold_train, yfold_train, Xfold_val, yfold_val])
                inputs.append(model_params)

                # Run the model
                m = model(*inputs)

                # Record the history of the run for this fold
                train_loss_history[str(j)] = m.history.history['loss']
                train_accuracy_history[str(j)] = m.history.history['accuracy']
                val_loss_history[str(j)] = m.history.history['val_loss']
                val_accuracy_history[str(j)] = m.history.history['val_accuracy']
                n_epochs.append(len(train_loss_history[str(j)]))

            # Update the metrics at the end of each configuration
            def mean_metric(metrics, n_epochs):
                df = pd.DataFrame([metrics[k] for k in metrics])
                df = df.sum(axis=0) / (len(df) - df.isna().sum())
                return(df.values)

            records['configs'][str(i)] = self.configs[i]
            records['mean_train_losses_history'][str(i)] = mean_metric(train_loss_history, n_epochs)
            records['mean_train_accuracies_history'][str(i)] = mean_metric(train_accuracy_history, n_epochs)
            records['mean_val_losses_history'][str(i)] = mean_metric(val_loss_history, n_epochs)
            records['mean_val_accuracies_history'][str(i)] = mean_metric(val_accuracy_history, n_epochs)

            # Scores are considered best ones if their corresponding loss is the best one
            if min(records['mean_val_losses_history'][str(i)]) < records['best_mean_val_loss']:
                argmin = np.argmin(records['mean_val_losses_history'][str(i)])
                records['best_mean_val_loss'] = records['mean_val_losses_history'][str(i)][argmin]
                records['best_mean_train_accuracy'] = records['mean_train_accuracies_history'][str(i)][argmin]
                records['best_mean_val_accuracy'] = records['mean_val_accuracies_history'][str(i)][argmin]
                records['best_config'] = self.configs[i]
                records['best_config_n'] = i
                records['best_epoch_n'] = argmin

                print(f"\n Currently best mean val_loss: {records['best_mean_val_loss']}")
                print(f"\n Currently best mean val_accuracy: {records['best_mean_val_accuracy']}")

                if run == None:
                    pass
                else:
                    run.log('current_best_mean_val_loss', records['best_mean_val_loss'])
                    run.log('current_best_mean_train_accuracy', records['best_mean_train_accuracy'])
                    run.log('current_best_mean_val_accuracy', records['best_mean_val_accuracy'])
                    run.log('current_best_config_n', records['best_config_n'])
                    run.log('current_best_epoch_n', records['best_epoch_n'])

        # log_list the history of the best configuration
        if run == None:
            pass
        else:
            best_config_n = records['best_config_n']
            run.log_list('best_config', records['configs'][str(best_config_n)])
            run.log_list('best_mean_train_losses_history', records['mean_train_losses_history'][str(best_config_n)])
            run.log_list('best_mean_train_accuracies_history', records['mean_train_accuracies_history'][str(best_config_n)])
            run.log_list('best_mean_val_losses_history', records['mean_val_losses_history'][str(best_config_n)])
            run.log_list('best_mean_val_accuracies_history', records['mean_val_accuracies_history'][str(best_config_n)])

        return(m, records)


"""
#Test add_parameters and add_config

del sim
sim = Sim()
sim.add_parameters(("p2",45),("p3",8))
print(f"\nAprès ajout des paramètres p2,45 et p3,8:\n{sim.parameters}")
sim.add_config(p55=58,p2=88)
print(f"\nAprès ajout des configs p55,58 et p2,88:\n{sim.configs}")
sim.add_config(p2=44,p3=57)
print(f"\nAprès ajout des configs p2,44 et p3,57:\n{sim.configs}")
sim.add_parameters(("p4",0),("p5",1))
print(f"\nAprès ajout des paramètres p4,0 et p5,1:\n{sim.parameters}")
print(f"\nAprès ajout des paramètres p4,0 et p5,1:\n{sim.configs}")
"""

""" 
#Test append_dict2list      
sim=Sim()   
l_in=[{"p1":1, "p2":4},{"p1":1, "p2":5},{"p1":1, "p2":6},\
      {"p1":2, "p2":4},{"p1":2, "p2":5},{"p1":2, "p2":6}] 
d_in={"p3":[6,7,8]} 
l_out = sim.append_dict2list(l_in, d_in)
print(l_out)
"""

"""
#Test dict2list

sim=Sim()
if 'd' in locals():
    del(d)
if 'd1' in locals():
    del(d1)
d_in={"p1":[1,2], "p2":[4,5], "p3":[6,7,8]}
l_out = sim.dict2list(d_in)
"""

"""
#Test initialize_model_inputs

sim=Sim()

l=("param1", "param2", "param3")
sim.initialize_model_inputs(*l)
print(sim.model_inputs)
"""


"""
#Test the integration of the functions

s = Sim()

train_set_x = 1
train_set_y = 2
test_set_x = 3
test_set_y = 4

s.add_parameters(('train_set_x', train_set_x),\
                 ('train_set_y', train_set_y),\
                 ('test_set_x', test_set_x),\
                 ('test_set_y', test_set_y),\
                 ('num_iterations', 1000),\
                 ('learning_rate', 0.005))

s.add_configs({"num_iterations":[500, 1500],\
               "learning_rate": [0.005, 0.01]})
    
s.initialize_model_inputs('train_set_x','train_set_y','test_set_x',\
                 'test_set_y','num_iterations','learning_rate')

def model(train_set_x, train_set_y, test_set_x,\
          test_set_y, num_iterations, learning_rate):
    return(f"num_iterations = {num_iterations} et learning rate = {learning_rate}")

s.run_model(model)
"""


    
    

    