# importing necessary modules
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import tensorflow as tf
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import  Dense, Input, Concatenate
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.metrics import AUC, Precision, Recall, TrueNegatives
from result_saver import save_results
from roc_curve_generator import save_roc_curve
import numpy as np
import pandas as pd
import math


# input

    
dense_archs =   [[64,32,16,8,4],
                [32,16,8,4],
                [16,8,4],
                [8,4],
                [16,16,16],
                [8,8,8],
                [4,4,4],
                [64,8],
                [32,8],
                [16,8],
                [64,4],
                [32,4],
                [16,4]] 

batch_size = 10
random_seed = 21

# dataset
version_number = '1000_0_0_without_imaging'

file_name_to_record_results = '*****/hipfracture/python/results/results.csv'

X_train = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/X_train_imputated_v'+version_number+'.pkl')
y_train = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/y_train_v'+version_number+'.pkl')

X_val = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/X_val_imputated_v'+version_number+'.pkl')
y_val = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/y_val_v'+version_number+'.pkl')

X_train = X_train.to_numpy()
X_val = X_val.to_numpy()
from time import time
for i,dense_arch in enumerate(dense_archs):
    random_seed = 20
    for j in range(5):
        random_seed += j
        new_experiment_name = 'experiment_v1000_3_'+str(i)+'_dataset_' + version_number+str(int(time()))
        
        # fully connected neuron architecture
        
        str_rep_dense_arch = ' - '.join([str(k) for k in dense_arch])
        
        experiment_notes = 'structured data on neural network, FC arch --> '+str_rep_dense_arch
        
        
        def build_model():
              
            input_structured = Input(shape=(99,))
            x = input_structured
            for nr_of_neurons in dense_arch:
                x = Dense(nr_of_neurons, activation='relu')(x)
                
            outputs = Dense(1, activation='sigmoid')(x)
            model = Model(inputs=input_structured, outputs=outputs)
            return model
        
        early_stop = EarlyStopping(monitor='val_auc', mode = 'max', patience=10, verbose=1, min_delta=1e-4, restore_best_weights = True)
        reduce_lr = ReduceLROnPlateau(monitor='val_auc', mode = 'max', factor=0.05, patience=5, verbose=1, min_delta=1e-4)
        csv_logger = CSVLogger('*****/hipfracture/csv_logs/log_' + new_experiment_name + ".csv", append=True, separator=';')
        new_filepath = '*****/hipfracture/model_weights/30day_mortality_learning/weights_' + new_experiment_name + ".hdf5"
        checkpointer = ModelCheckpoint(filepath=new_filepath, verbose=1, save_best_only=True, monitor='val_auc', mode='max')
        
                
        model = build_model()
        
        
        METRICS = [Precision(name='precision'),
                   Recall(name='recall'),
                   AUC(name='auc')]
        
        model.compile(loss='binary_crossentropy', optimizer=Adam(1e-3), metrics=METRICS)
        print("Experiment name {}".format(new_experiment_name))
        
        train = tf.data.Dataset.from_tensor_slices((X_train,y_train))
        train = train.repeat().shuffle(random_seed).batch(10)
        
        val = tf.data.Dataset.from_tensor_slices((X_val,y_val))
        val = val.repeat().batch(10)
                    
        history = model.fit_generator(
            train,
            steps_per_epoch = math.ceil(len(X_train) /batch_size),
            epochs=100,
            validation_data = val,
            validation_steps = math.ceil(len(X_val) /batch_size),
            callbacks = [early_stop, reduce_lr, csv_logger, checkpointer])
        
        # inference using trained model
        
        y_predict_scores = model.predict(val, verbose = 1, steps = int(np.ceil(len(X_val) /batch_size)))
        
        # default threshold = 0.5
        
        y_predict = y_predict_scores >= 0.5
        
        y_true = y_val.astype(int)
        
        
        
        ##############################################################################
        
        # saving validation(inference) results
        
        save_results(file_name_to_record_results, y_true = y_true, y_predict = y_predict, 
                     y_predict_scores = y_predict_scores,
                     experiment_name = new_experiment_name,
                     experiment_notes = experiment_notes, classifier = 'FC nn')
