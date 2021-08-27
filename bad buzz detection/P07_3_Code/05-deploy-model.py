ROOT_DIR = r"D:\Data\Google Drive\Openclassrooms\P7\Projet"
SRC_DIR = ROOT_DIR + r"\src"

import sys
sys.path.append(SRC_DIR)
import config

import importlib
importlib.reload(config)

from azureml.core import Workspace, Environment
from azureml.core.model import InferenceConfig, Model
from azureml.core.webservice import AciWebservice

#Retrieve the model name
model_name = config.model_name

# Load the workspace
ws = Workspace.from_config()
print("\nWorkspace loaded")

# Clean-up
# Delete services with the same name
if config.model_name in ws.webservices:
    ws.webservices[config.model_name].delete()
print('\nClean-up done')

# Configure an environment
env = Environment.from_conda_specification(name='P7-env',
                                           file_path='./.azureml/P7-env.yml')
print("\nEnvironment configured")

# Combine the script and environment in an InferenceConfig
inference_config = InferenceConfig(source_directory='./src',
                                   entry_script='entry_script.py',
                                   environment=env)
print("\nInference configured") 

# Define what compute target to use
compute_target = ws.compute_targets['cpu-cluster']
print("\nCompute target defined")

# Define the deployment configuration which sets the target 
# specific compute specification for the containerized deployement
deployment_config = AciWebservice.deploy_configuration(cpu_cores = 1, memory_gb = 1)
print("\nDeployment configured")

# Deploy the model
model = ws.models[model_name]
service = Model.deploy(workspace=ws,
                       name = model_name,
                       models = [model],
                       inference_config = inference_config,
                       deployment_config = deployment_config,
                       deployment_target = None)
service.wait_for_deployment(show_output = True)
print('\nDeployment done')