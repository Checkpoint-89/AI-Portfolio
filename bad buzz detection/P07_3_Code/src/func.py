import pandas as pd
import numpy as np
import re
import string
import nltk
from collections import Counter
from sklearn.metrics import accuracy_score

def sample_tweets(df, n):
    # Return a sample of length n of tweets and their sentiment label
    # negative -> 0
    # positive -> 1
    tweets = []
    sentiments = []

    if n%2 == 0:
        n1 = int(n/2)
        n2 = n1

    if n%2 == 1:
        n1 = int(n/2)
        n2 = n1 +1

    idx = df['label'] == 0
    tweets.extend(df.loc[idx, 'tweet'][0:n1])
    sentiments.extend(np.full(n1,0).tolist())

    idx = df['label'] == 4
    tweets.extend(df.loc[idx, 'tweet'][0:n2])
    sentiments.extend(np.full(n2,1).tolist())

    return(tweets, sentiments)

def aml_sa_extract_prediction(result):
    pred = []
    feat = []
    for i in range(len(result)):
        try:
            pred.append(np.argmax([result[i].confidence_scores.negative,
                        result[i].confidence_scores.positive])),
            feat.append([result[i].confidence_scores.negative,
                        result[i].confidence_scores.neutral,
                        result[i].confidence_scores.positive])
        except:
            pred.append(-1)
            feat.append(-1,-1,-1)
    
    return(pred, feat)

def aml_sa_accuracy(labels, preds):
    preds_ = [x for x in preds if (x == 0 or x == 2)]
    dropped = [i for i in range(len(preds)) if (preds[i] == -1)]
    labels_ = np.array(labels)
    labels_ = np.delete(labels_, dropped)

    accuracy = accuracy_score(labels, preds)

    return(accuracy, len(dropped))

def preprocess(doc, stemming=False, lemmatization=False, pos_tag=False, sw=None, output='pp_doc'):
    # Return the preprocessed documents if output = 'pp_doc'
    # Return the tokenized prepocessed document if output = 'tk_doc'

    # Denoise 1
    doc = re.sub(r'http.\S+', '', doc) # remove urls
    doc = re.sub(r'@.\S+', '', doc) # remove usernames starting with @...
    doc = re.sub(r'#.\S+', '', doc) # remove hashtags starting with #...
    doc = re.sub(r'\d+', '', doc) # remove digits
    doc = re.sub(r'\s[a-zA-Z]\s', ' ', doc) # remove single letters (between spaces)

    # Tokenize data
    from nltk.tokenize import TweetTokenizer
    tknzr = TweetTokenizer(reduce_len=True) # reduce length of repeatedd letters to max 3
    tokens = tknzr.tokenize(doc)
    
    # Denoise 2
    punct = string.punctuation
    tokens = [t for t in tokens if t not in punct] # Remove (single) punctuation signs
    tokens = [t.lower() for t in tokens] # Lower capitalization

    # Normalize - Stemming
    from nltk.stem import PorterStemmer
    if stemming == True:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]
        
    # Normalize - Lemmatization
    from nltk.stem import WordNetLemmatizer 
    if lemmatization == True:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        
    # Filter POS
    if pos_tag == True:
        tokens = [w[0] for w in nltk.pos_tag(tokens) 
                       if (w[1][0:2] == 'NN')]
       #tokens = [w[0] for w in nltk.pos_tag(tokens) 
                      #if (w[1][0:2] == 'NN' or w[1][0:2] == 'VB')]

    # Remove stop-words if the argument sw is not None
    if sw is not None:
        tokens = [t for t in tokens if t not in sw]

    # Compute preprocessed document
    pp_doc = ' '.join(tokens) #PreProcessed documents

    if output == 'tk_doc':
        return(tokens)
    if output == 'pp_doc':
        return(pp_doc)

def reduce_vocab_dim(tk_corpus, doc_f, term_f, min_df, max_df):
    # Remove words with a document frequency outside [min_df, max_df] from
    # tk_corpus, doc_f, term_f

    l = len(tk_corpus)
    min_df = min_df * l
    max_df = max_df * l

    term_f_out = Counter()
    doc_f_out = Counter()

    tk_corpus_out = []
    for tk_doc in tk_corpus:
        tk_doc_out = [t for t in tk_doc
                      if (doc_f[t] >= min_df and doc_f[t] <= max_df)]
        tk_corpus_out.append(tk_doc_out)
        term_f_out.update(tk_doc_out)
        doc_f_out.update(set(tk_doc_out))

    return(tk_corpus_out, doc_f_out, term_f_out)