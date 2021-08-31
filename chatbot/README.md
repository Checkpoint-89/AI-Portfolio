# Chatbot: flight booking
> Keywords: git, Microsoft Bot Framework , LUIS, Azure 

A MVP of a flight reservation service using Microsoft Bot Framework. The Bot is integrated with Azure resources: 1) LUIS (Language Understanding) detects intents and entities in the utterances, 2) CosmosDB keeps track of the failed conversations for diagnostic and improvement purposes, 3) Azure telemetry service records relevant activity which can be monitored with Insights, analyzed through Kusto requests and triggers alarms.  


## Content of the folder:

P10_00_presentation.pdf: supporting slides for the presentation of the project.

P10_01_data_prep: notebook - exploration and cleaning of the data

P10_02_luis_train_deploy.ipynb: notebook - training and deployment of the language model (LUIS)

P10_03_performance_monitoring.pdf: monitoring rules

P10_04_src: source code, check the README file for further details
