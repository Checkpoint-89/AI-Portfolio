# Recommendation system
> Keywords: surprise, RecSys, Collaborative filtering, Serverless Azure Function

MVP of a recommendation model called by a mobile application. It provides a ranked list of books to the reader. The mobile calls a serverless function on Azure which retrieves recommendations from a CosmosdB. The recommendations are computed off-line. The slides of the project provide also architecture principles of the final product.

## Content of the folder:

P9_00_presentation.pdf: supporting slides for the presentation of the project.

P9_01_scripts/eda.ipynb: notebook - data exploration and cleaning.

P9_01_scripts/reco.ipynb: notebook - offline calculation of the recommendations.

[Not in github] P9_02_scripts/to_cosmosdb: recommendations generated and uploaded in CosmosDB.

P9_01_scripts/az function/HttpTrigger1/_init__.py: Azure function called by the mobile app.

[The mobile app was provided by a third party]