import logging

import azure.functions as func
from azure.cosmos import CosmosClient
from credentials import COSMOS_KEY
from credentials import COSMOS_URI

def main(req: func.HttpRequest) -> func.HttpResponse:
         
    logging.info('\nPython HTTP trigger function processed a request.')

    userID = req.params.get("userId")

    if userID == None:
        try:
            req_body = req.get_json()
            logging.info(f"req_body: {req_body}")
        except ValueError:
            logging.info(f"ValueError")
            pass
        else:
            userID = req_body.get('userId')

    logging.info(f'userID : {userID}')
    logging.info(f'userID type: {type(userID)}')

    if userID != None:

        # Initialize the Cosmos client
        endpoint = COSMOS_URI
        key = COSMOS_KEY

        # Connect to containers
        client = CosmosClient(endpoint, key)
        database = client.get_database_client('reco')
        contentbase = database.get_container_client('contentbase')
        collab = database.get_container_client('collab')

        # Query them in SQL
        query = 'SELECT c["0"],c["1"],c["2"],c["3"],c["4"],c["5"] FROM c WHERE c.userID=' + str(userID)    
        it_collab = collab.query_items(query,enable_cross_partition_query=True)
        it_contentbase = contentbase.query_items(query,enable_cross_partition_query=True)

        # Extract
        try:
            recos = list(next(it_collab).values())
            print(f"collab: {recos}")
        except StopIteration:
            recos = list(next(it_contentbase).values())
            print(f'contentbase: {recos}')

        return func.HttpResponse(f"{recos}")

    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )