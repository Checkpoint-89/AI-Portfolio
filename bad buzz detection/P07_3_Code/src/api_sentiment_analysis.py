from azure.ai.textanalytics import TextAnalyticsClient

def api_sentiment_analysis_example(client):

    documents = [
       "I had the best day of my life.",
       "This was a waste of my time. The speaker put me to sleep.",
       "No tengo dinero ni nada que dar...",
       "L'hôtel n'était pas très confortable. L'éclairage était trop sombre."
   ]
    response = client.analyze_sentiment(documents=documents)
    return(response)

def api_sentiment_analysis(client, documents):

    response = client.analyze_sentiment(documents=documents)
    return(response)