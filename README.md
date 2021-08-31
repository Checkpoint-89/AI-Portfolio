# Hands-on training in Artificial Intelligence

Already passionate about Artificial Intelligence at University when I studied Signal & Image processing and Pattern Recognition, I want to code and deploy algorithms again. As a taste for entrepreneurship brought me into business for a while, I strengthened my technical expertise and practice over the last year (2020-21) and am now eager to discuss opportunities to develop new technologies and products!

This repository presents some of the hands-on work of my 12 months training.  

## bad buzz detection  
> Keywords: nltk, gensim, Tensorflow, Azure ML, NLP, vectorization, embeddings, CNN, LSTM 

A project about sentiment analysis using Azure ML to manage the training runs of the models and implementing a home made GridSearchCV within Keras. This project also tests the drag and drop Azure ML designer. 


## chatbot 
> Keywords: git, Microsoft Bot Framework , LUIS, Azure 

A MVP of a flight reservation service using Microsoft Bot Framework. The Bot is integrated with Azure resources: 1) LUIS (Language Understanding) detects intents and entities in the utterances, 2) CosmosDB keeps track of the failed conversations for diagnostic and improvement purposes, 3) Azure telemetry service records relevant activity which can be monitored with Insights, analyzed through Kusto requests and triggers alarms.


## client scoring  
> Keywords: Classification, Feature engineering, ML model training, Explainability, Performance measure, Hyperparameters optimization.

A model for the scoring of candidates to a bank credit. An application of standard ML classification algorithms. Regression, SVM and ensemble algorithms are compared. The performance is expressed with a business perspective, taking into account the balance between risks and opportunities and the explainability of the models.


## data exploration  
> Keywords: matplotlib, seaborn, plotly, Exploratory Data Analysis, Statistics, PCA, UI  

Provide, as a web interface, an exploratory tool to easely explore the OpenFoodFacts dataset. Requirements: using Ipywidget and Voila.


## image semantic segmentation
> Keywords: Tensorflow, Encoder-Decoder, Transfer learning, Fine tuning, VGG16, Unet, ResNet, PspNet

Explore various semantic segmentation algorithms on the CityScapes dataset. The selected model is integrated with Azure ML and a light web app is developped in flask for demonstration purposes.


## market segmentation
> Keywords: scikit-learn, Clustering, K-Means, DBSCAN, Agglomerative clustering

An application of clustering to the clients of an online retailer.

## recommendation system
> Keywords: surprise, RecSys, Collaborative filtering, Serverless Azure Function

MVP of a recommendation model called by a mobile application. It provides a ranked list of books to the reader. The mobile calls a serverless function on Azure which retrieves recommendations from a CosmosdB. The recommendations are computed off-line. The slides of the project provide also architecture principles of the final product.

## topic modelling
> Keywords: scikit-learn, nltk, Latent Dirichlet Allocation, Multi-Dimensional-Scaling, pyLDAVis

Evaluate the feasibility to assess insatisfaction topics within Yelp reviews:  
- What are the main insatisfaction topics ?  
- In one particular review: what is the client complaining about ?  

Provide a graphical answer to the above questions.



