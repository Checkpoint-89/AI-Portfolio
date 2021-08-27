# Update sys.path
import sys
ROOT_DIR = r"D:\Data\Google Drive\Openclassrooms\P7\Projet"
SRC_DIR = ROOT_DIR + r"\src"
sys.path.append(SRC_DIR)

import config
from azureml.core import Workspace, Experiment, Environment, Dataset, ScriptRunConfig

if __name__ == "__main__":

    # Name the model
    model_name = config.model_name

    # Load the workspace
    ws = Workspace.from_config()
    print("\nWorkspace loaded")

    # Get the default Azure Machine Learning datastore
    datastore = ws.get_default_datastore()
    print("\nDatastore created")

    # Instantiate an Azure Machine Learning dataset
    dataset = Dataset.File.from_files(path=[(datastore, 'datasets')])
    print("\nDataset instantiated")

    # Register the dataset
    dataset = dataset.register(workspace=ws,
                            name='data',
                            description='Données P7',
                            create_new_version=True)
    print("\nDataset registered")

    # Define what compute target to use
    compute_target = ws.compute_targets['gpu-cluster']
    print("\nCompute target defined")

    # Configure an environment
    env = Environment.from_conda_specification(name='P7-env-gpu',
                                            file_path='./.azureml/P7-env-gpu.yml')
    print("\nEnvironment configured")

    # Specify a GPU base image
    env.docker.enabled = True
    env.docker.base_image = 'mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04'
    print("\nGPU image specified")

    # Instantiate a script
    # Define the source directory, the main script, the compute target, the arguments
    src = ScriptRunConfig(source_directory='./src',
                        script='train.py',
                        compute_target=compute_target,
                        arguments=[
                                    '--data-folder', dataset.as_named_input('input').as_mount(),
                                    '--model-name', model_name,
                                    ],
                        )
    print("\nScript configured")

    # Update the script with the environment
    src.run_config.environment = env
    print("\nConfig updated with the GPU image")

    # Instantiate an experiment
    experiment = Experiment(workspace=ws, name='xp-tweets-2')
    print("\nExperiment instantiated")

    # Run an experiment
    run = experiment.submit(src)

    # Display the link to the experiment
    aml_url = run.get_portal_url()
    print("\nSubmitted to compute cluster. Click link below")
    print("")
    print(aml_url)