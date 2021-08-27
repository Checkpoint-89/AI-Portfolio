# Detecting bad buzz in social media  
> Keywords: nltk, gensim, Tensorflow, Azure ML, NLP, vectorization, embeddings, CNN, LSTM 

A project about sentiment analysis using Azure ML to manage the training runs of the models and implementing a home made GridSearchCV within Keras. This project also tests the drag and drop Azure ML designer. 


## Content of the folder:

P07_0_Presentation.pdf : supporting slides for the presentation of the project.

P07_1_BlogPost.pdf : blog article

P07_3_Code: code
---/script_config.py : config file  
---/znb-test-cognitive-service.ipynb: file that calls the Azure Cognitive Service: 'sentiment analysis'    
---/prototypage: notebook used to prototype the model    
---/01-create-workspace.py: create a workspace on Azure ML  
---/02-create-compute-cpu.py ou -gpu: CPU and GPU configuration on Azure ML  
---/03-upload-data.py : upload the dataset and the appropriate folder structure needed to train the model   
---/04-train-model.py : launch a training run on the compute  
---/05-deploy-model.py : deploy the model to be called by as an API  
---/06-consume-endpoint.py : test a call to the service   
---/src: source code of the modules to be trained  
------/see the blog post, §"Architecture logicielle et intégration dans Azure ML" for further details  