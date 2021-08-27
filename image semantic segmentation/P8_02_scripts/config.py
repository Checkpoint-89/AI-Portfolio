import importlib
import metrics
importlib.reload(metrics)
from metrics import dice, dice_loss
import numpy as np

run_eagerly = False

experiment_name = 'seg-2'

img_size = (384, 576)
IMAGE_ORDERING = 'channels_last'

num_classes = 8

weights_direct = np.array([0.10754258, 0.38708704, 0.21769261, 0.01759003, 0.14986737, 0.03492508, 0.01198436, 0.07331093])
weights_inverse = np.array([0.04520181, 0.01255821, 0.0223302 , 0.27635645, 0.03243614, 0.13918707, 0.4056219 , 0.06630824])
weights_units = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ])

model_type = 'unet_like' # unet_like vgg_unet resnet50_unet resnet50_pspnet

if model_type == 'unet_like':
    model_input_size = (224,224) #(448,448) #(224,224)
    model_output_size = (224,224) #(224,224) #(112,112)

if model_type == 'vgg_unet':
    model_input_size = (224,224) #(448,448) #(224,224)
    model_output_size = (112,112) #(224,224) #(112,112)

if model_type == 'resnet50_unet':
    model_input_size = (224,224)
    model_output_size = (112,112)

if model_type == 'resnet50_pspnet':
    model_input_size = (384, 576)
    model_output_size = (384, 576)

model_name = model_type + '_' + 'with_augm-entropy'

# 1
if model_name == model_type + '_' + 'wo_augm-entropy':
    trainable_base = False
    true_one_hot = False
    augmentation = False
    loss = "sparse_categorical_crossentropy"
    # optimizer = "rmsprop"
    optimizer = "adam"
    metrics= ['sparse_categorical_accuracy', dice(weights_units)]
    
# 2
if model_name == model_type + '_' + 'with_augm-entropy':
    trainable_base = False
    true_one_hot = False
    augmentation = True
    loss = "sparse_categorical_crossentropy"
    #optimizer = "rmsprop"
    optimizer = "adam"
    metrics= ['sparse_categorical_accuracy', dice(weights_units)]
    
# 3
if model_name == model_type + '_' + 'wo_augm-dice':
    trainable_base = False
    true_one_hot = False
    augmentation = False
    loss = dice_loss(weights_units)
    #optimizer = "rmsprop"
    optimizer = "adam"
    metrics= ['sparse_categorical_accuracy', dice(weights_units)]
    
# 4
if model_name == model_type + '_' + 'with_augm-dice':
    trainable_base = False
    true_one_hot = False
    augmentation = True
    loss = dice_loss(weights_units)
    #optimizer = "rmsprop"
    optimizer = "adam"
    metrics= ['sparse_categorical_accuracy', dice(weights_units)]

batch_size = 8
if model_type == 'unet_like':
    epochs = 50
else:
    epochs = 35
epochs_fine_tuning = 15
steps_per_epoch = None
validation_steps = None
learning_rate = 1e-3
learning_rate_fine_tuning = 0.0001
early_stopping_min_delta = 0.0001
early_stopping_patience = 3
lr_reduce_factor = 0.5
lr_reduce_min_delta = 0.001
lr_reduce_patience = 10
lr_reduce_min_lr = 1e-6



