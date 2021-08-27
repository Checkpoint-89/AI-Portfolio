# preprocess.py
import importlib

# Python
import re
from collections import Counter

# Data libraries
import pandas as pd
import numpy as np

# NLP
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from tensorflow.keras.preprocessing.sequence import pad_sequences

# My modules
import func
importlib.reload(func)

def preprocess_train(corpus_train, y_train, model_params):

    print("\n****************************************")
    print("Preprocess train set")
    print("****************************************")

    # Retrieve the model parameters
    stemming = model_params['preprocessing']['stemming']
    lemmatization = model_params['preprocessing']['lemmatization']
    pos_tag = model_params['preprocessing']['pos_tag']
    min_df = model_params['preprocessing']['min_df']
    max_df = model_params['preprocessing']['max_df']

    # Create a set of stop-words
    sw = set((tuple(nltk.corpus.stopwords.words('english'))))
    sw = set(func.preprocess(' '.join(list(sw)),
                    stemming=stemming,
                    lemmatization=lemmatization,
                    pos_tag=False,
                    sw=None,
                    output='tk_doc'))
    sw=[] # No stop words
    
    # Record model parameters
    model_params['preprocessing']['sw'] = sw

    # Tokenize
    tk_train_corpus = []
    term_f_train = Counter()
    doc_f_train = Counter() 
    for doc in corpus_train:
        tk_doc = func.preprocess(doc, stemming=stemming, lemmatization=lemmatization, pos_tag=pos_tag, sw=sw, output='tk_doc')
        tk_train_corpus.append(tk_doc)
        term_f_train.update(tk_doc)
        doc_f_train.update(set(tk_doc))
    print('Tokenization of the train set done')
    print('Vocabulary size after tokenization: %d' % len(term_f_train))

    # Remove rare & common words
    min_df_ = min_df / len(corpus_train)
    tk_train_corpus, doc_f_train, term_f_train = func.reduce_vocab_dim(tk_train_corpus, doc_f_train, term_f_train, min_df_, max_df)
    vocab_train = {k:v for v,k in enumerate(term_f_train.keys())}
    vocab_size_train = len(vocab_train)
    print('Vocabulary reduction done')
    print('Vocabulary size after reduction: %d' % vocab_size_train)

    # Record model parameters
    model_params['preprocessing']['vocab_train'] = vocab_train

    # Encode the docs
    tk_train_corpus_enc = []
    for tk_doc in tk_train_corpus:
        tk_doc_enc = [vocab_train[token] for token in tk_doc]
        tk_train_corpus_enc.append(tk_doc_enc)
    print('Encoding of the train set done')

    # Pad the encoded documents of the corpus
    tk_doc_max_len = max([len(tk_doc_enc) for tk_doc_enc in tk_train_corpus_enc])
    padded_corpus_train = pad_sequences(tk_train_corpus_enc, maxlen = tk_doc_max_len, padding='post')
    print('Padding of the train set done')

    print(f"Train size: {len(padded_corpus_train), len(y_train)}")

    # Record model parameters
    model_params['preprocessing']['tk_doc_max_len'] = tk_doc_max_len

    return(tk_train_corpus , padded_corpus_train)

def preprocess_test(corpus_test, y_test, model_params = dict()):
    #################################################
    # VALIDATION & TEST SET 

    print("\n****************************************")
    print("Preprocess test set")
    print("****************************************")

    # Retrieve the model parameters
    stemming = model_params['preprocessing']['stemming']
    lemmatization = model_params['preprocessing']['lemmatization']
    pos_tag = model_params['preprocessing']['pos_tag']
    sw = model_params['preprocessing']['sw']
    vocab_train = model_params['preprocessing']['vocab_train']
    tk_doc_max_len = model_params['preprocessing']['tk_doc_max_len']
    val_test_ratio = model_params['train_test_split']['val_test_ratio']

    # Tokenize
    tk_test_corpus = []
    for doc in corpus_test:
        tk_doc_test = func.preprocess(doc, stemming=stemming, lemmatization=lemmatization, pos_tag=pos_tag, sw=sw, output='tk_doc')
        tk_test_corpus.append(tk_doc_test)
    print('Tokenization of the test set done')

    # Encode the docs
    tk_test_corpus_enc = []
    for tk_doc in tk_test_corpus:
        tk_doc_enc = [vocab_train[token] for token in tk_doc if token in vocab_train]
        tk_test_corpus_enc.append(tk_doc_enc)
    print('Encoding of the test set done')

    # Pad the encoded documents of the corpus
    padded_corpus_test = pad_sequences(tk_test_corpus_enc, maxlen = tk_doc_max_len, padding='post')
    print('Padding of the test set done')

    print(f"Test size: {len(padded_corpus_test), len(y_test)}")

    print("\n****************************************")
    print("Split test set further into validation and test sets")
    print("****************************************")
    # Split into val and test set
    val_test_ratio = val_test_ratio
    idx = round(val_test_ratio * len(padded_corpus_test))
    padded_corpus_val = padded_corpus_test[:idx]
    y_val = y_test[:idx]

    padded_corpus_test = padded_corpus_test[idx:]
    y_test_ = y_test[idx:]

    print('Split into dev and test set done')
    print(f"Validation size: {len(padded_corpus_val), len(y_val)}")
    print(f"Test size: {len(padded_corpus_test), len(y_test_)}")

    return(padded_corpus_val, padded_corpus_test, y_val, y_test_)

def preprocess(corpus_train, corpus_test, y_train, y_test, val_test_ratio = 0.5, model_params = dict()):

    #################################################
    # PREPROCESS TRAIN AND TEST SETS
    #################################################

    # Create a set of stop-words
    sw = set((tuple(nltk.corpus.stopwords.words('english'))))
    sw = set(func.preprocess(' '.join(list(sw)),
                    stemming=stemming,
                    lemmatization=lemmatization,
                    pos_tag=False,
                    sw=None,
                    output='tk_doc'))
    sw=[] # No stop words

    #################################################
    # TRAIN SET

    # Record model parameters of the preprocessing in model_param
    model_params['preprocessing'] = dict()

    # Tokenize
    tk_train_corpus = []
    term_f_train = Counter()
    doc_f_train = Counter() 
    for doc in corpus_train:
        tk_doc = func.preprocess(doc, stemming=stemming, lemmatization=lemmatization, pos_tag=pos_tag, sw=sw, output='tk_doc')
        tk_train_corpus.append(tk_doc)
        term_f_train.update(tk_doc)
        doc_f_train.update(set(tk_doc))
    print('\nTokenization of the train set done')
    print('Vocabulary size after tokenization: %d' % len(term_f_train))
    
    # Record model parameters
    model_params['preprocessing']['stemming'] = stemming
    model_params['preprocessing']['lemmatization'] = lemmatization
    model_params['preprocessing']['pos_tag'] = pos_tag
    model_params['sw'] = sw

    # Remove rare & common words
    min_df_ = min_df / len(corpus_train)
    tk_train_corpus, doc_f_train, term_f_train = func.reduce_vocab_dim(tk_train_corpus, doc_f_train, term_f_train, min_df_, max_df)
    vocab_train = {k:v for v,k in enumerate(term_f_train.keys())}
    vocab_size_train = len(vocab_train)
    print('Vocabulary reduction done')
    print('Vocabulary size after reduction: %d' % vocab_size_train)

    # Record model parameters
    model_params['preprocessing']['vocab_train'] = vocab_train

    # Encode the docs
    tk_train_corpus_enc = []
    for tk_doc in tk_train_corpus:
        tk_doc_enc = [vocab_train[token] for token in tk_doc]
        tk_train_corpus_enc.append(tk_doc_enc)
    print('Encoding of the train set done')

    # Pad the encoded documents of the corpus
    tk_doc_max_len = max([len(tk_doc_enc) for tk_doc_enc in tk_train_corpus_enc])
    padded_corpus_train = pad_sequences(tk_train_corpus_enc, maxlen = tk_doc_max_len, padding='post')
    print('Padding of the train set done')

    # Record model parameters
    model_params['preprocessing']['tk_doc_max_len'] = tk_doc_max_len

    #################################################
    # VALIDATION & TEST SET 

    # Tokenize
    tk_test_corpus = []
    for doc in corpus_test:
        tk_doc_test = func.preprocess(doc, stemming=stemming, lemmatization=lemmatization, pos_tag=pos_tag, sw=sw, output='tk_doc')
        tk_test_corpus.append(tk_doc_test)
    print('\nTokenization of the test set done')

    # Encode the docs
    tk_test_corpus_enc = []
    for tk_doc in tk_test_corpus:
        tk_doc_enc = [vocab_train[token] for token in tk_doc if token in vocab_train]
        tk_test_corpus_enc.append(tk_doc_enc)
    print('Encoding of the test set done')

    # Pad the encoded documents of the corpus
    padded_corpus_test = pad_sequences(tk_test_corpus_enc, maxlen = tk_doc_max_len, padding='post')
    print('Padding of the test set done')

    # Split into val and test set
    val_test_ratio = val_test_ratio
    idx = round(val_test_ratio * len(padded_corpus_test))
    padded_corpus_val = padded_corpus_test[:idx]
    y_val = y_test[:idx]

    padded_corpus_test = padded_corpus_test[idx:]
    y_test = y_test[idx:]

    print('\nSplit into dev and test set done')
    print(f"Train size: {len(padded_corpus_train), len(y_train)}")
    print(f"Validation size: {len(padded_corpus_val), len(y_val)}")
    print(f"Test size: {len(padded_corpus_test), len(y_test)}")

    return(vocab_train, tk_doc_max_len, tk_train_corpus, padded_corpus_train, padded_corpus_val, padded_corpus_test, y_val, y_test)
