import config
import numpy as np
import pickle

embedding_output_dim = config.embedding_output_dim

def compute_dummy_embedding(embedding_dir, vocab_train):
    embedding_dict = {}
    for w,_ in vocab_train.items():
        embedding_dict[w] = np.random.randn(embedding_output_dim)

    # Save the embedding
    with open(embedding_dir + '/dummy.p', 'wb') as f:
        pickle.dump(embedding_dict, f)

    print("Shape of the embedding dict",(len(embedding_dict), len(list(embedding_dict.values())[0])))

def load_glove_embedding(data_dir, embedding_dir, vocab_train):
	# Load the whole embedding into memory
	embedding_index = dict()
	f = open(data_dir + r'/glove.twitter.27B.200d.txt', encoding='utf8')
	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.array(values[1:], dtype='float32')
		embedding_index[word] = coefs
	f.close()
	print('Loaded %s word vectors.' % len(embedding_index))

	# Sample in word of vocab_train
	embedding_dict = {}
	na_pretrained_glove = []
	for word, _ in vocab_train.items():
		try:
			embedding_dict[word] = embedding_index[word]
		except KeyError:
			na_pretrained_glove.append(word)

	# Save the embedding
	with open(embedding_dir + '/pretrained_glove.p', 'wb') as f:
		pickle.dump(embedding_dict, f)

	print("Shape of the embedding dict",(len(embedding_dict), len(list(embedding_dict.values())[0])))
	print("Words of the vocab NA in the embedding: ", len(na_pretrained_glove))

def compute_w2v_embedding(embedding_dir, tk_train_corpus, vocab_train):

    from gensim.models.word2vec import Word2Vec
	
    # Train word2vec model
    model = Word2Vec(tk_train_corpus, size=embedding_output_dim, window=5, workers=8, min_count=1)

	# Sample in word of vocab_train
    embedding_dict = {}
    na_learned_w2v = []
    for word, _ in vocab_train.items():
	    try:
		    embedding_dict[word] = model.wv[word]
	    except KeyError:
		    na_learned_w2v.append(word)

	# Save the embedding
    with open(embedding_dir + '/learned_w2v.p', 'wb') as f:
	    pickle.dump(embedding_dict, f)

    print("Shape of the embedding dict",(len(embedding_dict), len(list(embedding_dict.values())[0])))
    print("Words of the vocab NA in the embedding: ", len(na_learned_w2v))


def load_w2v_embedding(data_dir, embedding_dir, vocab_train):

	from gensim.models import KeyedVectors

	filename = data_dir + r'/GoogleNews-vectors-negative300.bin'
	model = KeyedVectors.load_word2vec_format(filename, binary=True)

	# Create a weight matrix for words in training docs
	embedding_dict = {}
	na_pretrained_w2v = []
	for word, _ in vocab_train.items():
		try:
			embedding_dict[word] = model.wv[word]
		except KeyError:
			na_pretrained_w2v.append(word)

	# Save the embedding
	with open(embedding_dir + '/pretrained_w2v_300d.p', 'wb') as f:
		pickle.dump(embedding_dict, f)

	print("Shape of the embedding dict",(len(embedding_dict), len(list(embedding_dict.values())[0])))
	print("Words of the vocab NA in the embedding: ", len(na_pretrained_w2v))

def select_embedding(embedding_dir, vocab, embedding_type):


    print("\n****************************************")
    print("Select and load the embedding")
    print("****************************************")

    print(f"Selected embedding: {embedding_type}")

    if embedding_type == 'dummy':
        with open(embedding_dir + r'\dummy.p', 'rb') as f:
            word2vec = pickle.load(f)

    if embedding_type == 'learned_w2v':
        with open(embedding_dir + r'/learned_w2v.p', 'rb') as f:
            word2vec = pickle.load(f)

    if embedding_type == 'pretrained_w2v':
        with open(embedding_dir + r'/pretrained_w2v_300d.p', 'rb') as f:
            word2vec = pickle.load(f)

    if embedding_type == 'pretrained_glove':
        with open(embedding_dir + r'/pretrained_glove.p', 'rb') as f:
            word2vec = pickle.load(f)

    print("Original embedding loaded")

	# Create a weight matrix for words in training docs
    embedding_matrix = np.zeros((len(vocab), len(list(word2vec.values())[0])))
    for word, i in vocab.items():
        try:
            embedding_vector = word2vec[word]
        except KeyError:
            pass
        else:
            embedding_matrix[i] = embedding_vector

    print("Embedding reduced to the words in the vocabulary")
    print(f"Shape of the weights: {embedding_matrix.shape}")

    return(embedding_matrix)