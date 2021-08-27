import script_config as sc
from azureml.core import Workspace

ws = Workspace.create(name = 'WS-P7', # provide a name for the  workspace
                      subscription_id = sc.SUBSCRIPTION_KEY, # provide the subscription ID
                      resource_group = 'P7', # provide a resource group name
                      create_resource_group = False,
                      location='westeurope') # For example: 'westeurope' or 'eastus2' or 'westus2' or 'southeastasia'.

# write out the workspace details to a configuration file: .azureml/config.json
ws.write_config(path='.azureml')