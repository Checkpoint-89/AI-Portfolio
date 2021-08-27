# load.py

# Import reload
import importlib

# Import config file
import config

# Import librairies
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Reload libs
importlib.reload(config)

def load_and_split_docs(data_file, n_tweets=20000, test_size = 0.20):

    # Load the tweets
    print("\n****************************************")
    print("Load the data")
    print("****************************************")

    df = pd.read_csv(data_file,  encoding="ISO-8859-1", usecols=[0, 5], names=["label","tweet"])
    df.loc[df['label']==4,'label'] = 1

    print('Data loaded')
    print(f'Original dataset size: {df.shape}')

    # Sample tweets
    print("\n****************************************")
    print("Sample the data")
    print("****************************************")

    if float(1-n_tweets/len(df)) != 0.0:
        corpus, _, true_labels, _ = train_test_split(df['tweet'], df['label'], test_size=float(1-n_tweets/len(df)),
                                                     random_state=42, stratify = df['label'])
    else:
        corpus = df['tweet']
        true_labels = df['label']
    
    print('Data sampled')
    print(f'Sampled dataset size: ({corpus.shape}, {true_labels.shape})')

    # Train-test split
    print("\n****************************************")
    print("Train and test split")
    print("****************************************")

    corpus_train, corpus_test, y_train, y_test = train_test_split(corpus, true_labels, test_size=test_size,
                                                        random_state=42, stratify = true_labels)
    print('Train and test split done')
    print(f'Train set size: ({corpus_train.shape}, {y_train.shape})')
    print(f'Test set size: ({corpus_test.shape}, {y_test.shape})')


    return(corpus_train, corpus_test, y_train, y_test)