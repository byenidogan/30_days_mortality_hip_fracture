# importing necessary modules
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications import densenet
from tensorflow.python.keras.applications import xception
from tensorflow.python.keras.applications import inception_v3
from tensorflow.python.keras.applications import resnet

from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import  Dense, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.metrics import AUC, Precision, Recall, TrueNegatives
from tensorflow.python.keras.initializers import Constant
from tensorflow.python.keras import regularizers

from result_saver import save_results
from roc_curve_generator import save_roc_curve
import numpy as np
import pandas as pd
import math

##############################################################################
# general input area

# make it false to load model with the same experiment name
new_model = True

# only valid if new_model = False
old_experiment_name = ''

from time import time
# adding timestamp to experiment name, to not overwrite past experiments with same name
new_experiment_name = '30day_mortality_learning_from_chest_14_0_0_'+str(int(time()))

# select modality -> chest or hip
modality = "chest"

# select pre-trained convolutional model
classifier = xception.Xception
classifier_name = 'Xception'

# fully connected neuron architecture
dense_0_neurons = 256

dense_1_neurons = 8

initial_output_bias = False

adjusted_class_weights = False    

experiment_notes = 'full training, imagenet weights, FC arch --> '+\
    str(dense_0_neurons)+'-'+ str(dense_1_neurons)+', no class imbalance handling'

# required image dim is 224 for densenet
# required image dim is >75 for inception_v3, default is 299
# required image dim is 224 for resnet
# required image dim is >71 for Xception,  default is 299 

image_dim = 299
seed_value = 21
batch_size = 10
weights= 'imagenet'

# directory for images

# dont change unless you want to change the cohort
DIR = "*****/hipfracture/python/PNG_30Day_Mortality_HipFractures/30day_mortality_learning_from_imaging/"+modality+"/"


# end of general input area
##############################################################################

# callbacks to use during training
early_stop = EarlyStopping(monitor='val_auc', mode = 'max', patience=10, verbose=1, min_delta=1e-4, restore_best_weights = True)
reduce_lr = ReduceLROnPlateau(monitor='val_auc', mode = 'max', factor=0.05, patience=5, verbose=1, min_delta=1e-4)
csv_logger = CSVLogger('*****/hipfracture/csv_logs/log_' + new_experiment_name + ".csv", append=True, separator=';')
old_filepath = '*****/hipfracture/model_weights/30day_mortality_learning/weights_' + old_experiment_name + ".hdf5"
new_filepath = '*****/hipfracture/model_weights/30day_mortality_learning/weights_' + new_experiment_name + ".hdf5"
checkpointer = ModelCheckpoint(filepath=new_filepath, verbose=1, save_best_only=True, monitor='val_auc', mode='max')

# model builder function
def build_model(base_model = None,  output_bias=None, restore_weights=False, full_restore = False):

    if restore_weights:
        if full_restore:
            return base_model
        else:
            #chopped_model = Model(inputs = base_model.input, outputs = base_model.layers[-3])
            x = base_model.layers[-4].output
            x = Dense(dense_0_neurons, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
            #x = Dropout(0.25)(x)
            x = Dense(dense_1_neurons, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
            #x = Dropout(0.25)(x)
            outputs = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=base_model.input, outputs=outputs)
        return model
    else:

        # setting which layers are to be trained or freezed
        # dense169 last conv block 369
        # resnet152 last conv block 483
        # inceptionv3 last 2 block 249
        # xception last 2 block 116

        # for layer in base_model.layers[:]:
        #     layer.trainable = False
        
        for layer in base_model.layers[:]:
            layer.trainable = True
            
        
        # ARCHITECTURE AFTER Pre_Trained_network
        x = base_model.output
        x = Dense(dense_0_neurons, activation='relu')(x)
        x = Dense(dense_1_neurons, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid', bias_initializer=output_bias)(x)
        model = Model(inputs=base_model.input, outputs=outputs)
        return model

##############################################################################



METRICS = [Precision(name='precision'),
           Recall(name='recall'),
           AUC(name='auc'),
           TrueNegatives(name='true_negatives')]


if new_model:
    base_model = classifier(input_shape=(image_dim, image_dim, 3),
                                      weights=weights,
                                      include_top=False,
                                      pooling='avg')

    model = build_model(base_model = base_model, output_bias=initial_bias)
    model.compile(loss='binary_crossentropy', optimizer=Adam(1e-3), metrics=METRICS)

else:
    model = build_model(load_model(old_filepath), initial_bias, restore_weights= True, full_restore= False)
    model.compile(loss='binary_crossentropy', optimizer=Adam(1e-3), metrics=METRICS)
    
print("Experiment name {}".format(new_experiment_name))



# data generators
    
# data augmentation for training images
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    channel_shift_range=10,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')

# no augmentation for inference
test_datagen = ImageDataGenerator()


train_generator = train_datagen.flow_from_directory(
    directory=DIR + 'train',
    target_size=(image_dim, image_dim),
    batch_size=batch_size,
    class_mode="binary",
    interpolation="bicubic",
    shuffle=True,
    seed=seed_value)

validation_generator = test_datagen.flow_from_directory(
    directory=DIR + 'val',
    target_size=(image_dim, image_dim),
    batch_size=batch_size,
    class_mode="binary",
    interpolation="bicubic",
    shuffle=False,
    seed=seed_value)

test_generator = test_datagen.flow_from_directory(
    directory=DIR + 'test',
    target_size=(image_dim, image_dim),
    batch_size=batch_size,
    class_mode="binary",
    interpolation="bicubic",
    shuffle=False,
    seed=seed_value)


history = model.fit_generator(
    train_generator,
    steps_per_epoch=int(math.ceil(train_generator.samples / batch_size)),
    epochs=100,
    validation_data=validation_generator,
    validation_steps=int(math.ceil(validation_generator.samples / batch_size)),
    callbacks=[early_stop, reduce_lr, csv_logger, checkpointer])

##############################################################################

# saving validation results

filenames = validation_generator.filenames
nb_samples = len(filenames)

y_predict_scores = model.predict(validation_generator,verbose = 1)
y_predict = y_predict_scores >= 0.5
y_true = validation_generator.classes
print(validation_generator.class_indices)

save_roc_curve(experiment_name = new_experiment_name, y_true = y_true, y_predict = y_predict )

save_results('*****/hipfracture/python/results/results.csv', y_true = y_true, y_predict = y_predict, 
             y_predict_scores = y_predict_scores,
             experiment_name = new_experiment_name,
             experiment_notes = experiment_notes,classifier = classifier_name)

##############################################################################

# saving test results
experiment_notes = 'full training, imagenet weights, FC arch --> '+\
    str(dense_0_neurons)+'-'+ str(dense_1_neurons)+', no class imbalance handling, testing'


filenames = test_generator.filenames
nb_samples = len(filenames)

y_predict_scores = model.predict(test_generator,verbose = 1)
y_predict = y_predict_scores >= 0.5
y_true = test_generator.classes
print(test_generator.class_indices)

save_roc_curve(experiment_name = experiment_name, y_true = y_true, y_predict = y_predict )

save_results('*****/results/results.csv', y_true = y_true, y_predict = y_predict, 
             y_predict_scores = y_predict_scores,
             experiment_name = old_experiment_name,
             experiment_notes = experiment_notes,classifier = classifier_name)

