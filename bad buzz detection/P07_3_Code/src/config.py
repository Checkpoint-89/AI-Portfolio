# Reload
import importlib

# Imports
import sys

# Constants
SUBSCRIPTION_KEY = "KEY_HERE"
COGNITIVE_SERVICE_KEY = "KEY_HERE"
SENTIMENT_ANALYSIS_ENDPOINT = "https://api-sentiment-analysis.cognitiveservices.azure.com/"

########################################
# Training parameters
########################################

# Name of the experience
model_name = 'cnn_lstm_full_1M6'

# Size of the train, validation and test sets
n_samples = 1600000 # total number of samples: train + val + test sets
test_size = 0.05 # total samples for the val + test sets
val_test_ratio = 0.5 # must be > 0 ; ratio of the test_size that is allocated to the dev set

# Preprocessing
stemming = False
lemmatization = True
pos_tag = False
max_df= 1.0 # Max freq of documents in which a word occurs
min_df = 10 # Min munber of documents in which a word occurs

# Embedding
embedding_type = 'pretrained_glove'  # dummy learned_w2v pretrained_w2v  pretrained_glove
embedding_output_dim = 200
embedding_trainable = True

# Model
model_type = 'lstm+cnn' # base conv lstm lstm+cnn

# Model hyperparameters selection
n_folds = 3

early_stopping_min_delta = 0.001
early_stopping_patience = 3

lr_reduce_factor = 0.5
lr_reduce_min_delta = 0.005
lr_reduce_patience = 3
lr_reduce_min_lr = 0.00001

epochs = [1]
batch_size = [256]

learning_rate = [0.01]

dropout_rate = [0.0, 0.2, 0.5] #[0.0, 0.2, 0.5]
l2_reg = [0.0, 0.0001, 0.01, 1.0] #[0.0, 0.0001, 0.01, 1.0]

# Simulation 

# Pipeline steps
compute_bow = False
compute_dummy_embedding = False
load_glove_embedding = False
compute_w2v_embedding = False
load_w2v_embedding = False