import argparse
from azureml.core import Workspace
from azureml.core.webservice import Webservice
import json
import requests

parser = argparse.ArgumentParser()
parser.add_argument('data_list', type=lambda x: x, nargs='+')
args = parser.parse_args()
data_list = args.data_list

# Format the data
data = {"data": data_list}

print(data)

input_data = json.dumps(data)
print("\nData formatted")

# Connect to workspace
ws = Workspace.from_config()
print("\nWorkspace connected")

# Connect to webservice
webservice = Webservice(workspace=ws, name='model-1')
scoring_uri = webservice.scoring_uri
print("\nWebservice connected")

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
# headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print('\nRequest done')

print('\nResponse:')
print(resp.text)