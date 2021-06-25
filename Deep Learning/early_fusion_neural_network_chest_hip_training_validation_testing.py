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
import tensorflow as tf
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import  Dense, Input, Concatenate
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.metrics import AUC, Precision, Recall, TrueNegatives
from tensorflow.python.keras.initializers import Constant
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from result_saver import save_results
from roc_curve_generator import save_roc_curve
import numpy as np
import pandas as pd
import math
from sys import argv

from time import time
# input


def two_image_generator(generator,
                        dataframe,
                        directory_1,
                        directory_2,
                        batch_size,
                        x_1_col = 'filename',
                        x_2_col = 'filename',
                        y_col = None,
                        class_mode = 'binary',
                        shuffle = False,
                        img_size1 = (224, 224),
                        img_size2 = (224,224)):
    
    gen1 = generator.flow_from_dataframe(dataframe = dataframe,
                                        directory = directory_1,
                                        x_col = x_1_col,
                                        y_col = y_col,
                                        target_size = img_size1,
                                        class_mode = class_mode,
                                        batch_size = batch_size,
                                        shuffle = shuffle,
                                        seed = seed_value)

    gen2 = generator.flow_from_dataframe(dataframe = dataframe,
                                        directory = directory_2,
                                        x_col = x_2_col,
                                        y_col = y_col,
                                        target_size = img_size2,
                                        class_mode = class_mode,
                                        batch_size = batch_size,
                                        shuffle = shuffle,
                                        seed = seed_value)
    
    while True:
        X1i = gen1.next()
        X2i = gen2.next()
        if y_col:
            yield [X1i[0], X2i[0]], X1i[1]  #X1i[1] is the label
        else:
            yield [X1i, X2i]
                            
# model builder function
def build_model(base_model_chest = None, base_model_hip = None):
    # setting which layers are to be trained or freezed
    # dense169 last conv block 369
    # resnet152 last conv block 483
    # inceptionv3 last 2 block 249
    # xception last 2 block 116
    for layer in base_model_chest.layers[:]:
        layer.trainable = True
        layer._name = 'chest_'+layer.name
        
    for layer in base_model_hip.layers[:483]:
        layer.trainable = False
        layer._name = 'hip_'+layer.name
    
    for layer in base_model_hip.layers[483:]:
        layer.trainable = True
        layer._name = 'hip_'+layer.name
    
    # ARCHITECTURE AFTER Pre_Trained_network
    x = base_model_chest.output
    x = Dense(256, activation='relu')(x)
    x = Dense(8, activation='relu')(x)
    
    y = base_model_hip.output
    y = Dense(256, activation='relu')(y)
    y = Dense(8, activation='relu')(y)
    
    concatenated = Concatenate()([x, y])

    z = Dense(4, activation='relu')(concatenated)
    outputs = Dense(1, activation='sigmoid')(z)
    model = Model(inputs=[base_model_chest.input,base_model_hip.input], outputs=outputs)
    return model

##############################################################################

# required image dim is 224 for densenet
# required image dim is >75 for inception_v3, default is 299
# required image dim is 224 for resnet
# required image dim is >71 for Xception,  default is 299 


image_dim_chest = 299
image_dim_hip = 224

seed_value = 21
batch_size = 10
weights= 'imagenet'

# make it false to load model with the same experiment name
new_model = True
if new_model:
    experiment_name = '30day_mortality_learning_from_chest_and_hip_1000_30_0_'+str(int(time()))
else:
    script, experiment_name = argv

# directory for images
DIR_chest = "*****/hipfracture/python/PNG_30Day_Mortality_HipFractures/30day_mortality_learning_from_imaging/chest/production/all"
DIR_hip = "*****/hipfracture/python/PNG_30Day_Mortality_HipFractures/30day_mortality_learning_from_imaging/hip/production/all"

version_number = '******'



classifier = 'Xception - ResNet'

experiment_notes = 'learning from chest and hip simultaneously, from imagenet weights'

file_name_to_record_results = '*****/hipfracture/python/results/results.csv'
filepath = '*****/hipfracture/model_weights/30day_mortality_learning/weights_' + experiment_name + ".hdf5"

X_train = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/X_train_imputated_v'+version_number+'.pkl')
X_val = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/X_val_imputated_v'+version_number+'.pkl')
X_test = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/X_test_imputated_v'+version_number+'.pkl')

y_val = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/y_val_v'+version_number+'.pkl')
y_test = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/y_test_v'+version_number+'.pkl')


print ('X_train shape = ', X_train.shape)
print ('X_val shape = ', X_val.shape)
print ('X_test shape = ', X_test.shape)

# image reference dataset generation

df = pd.read_pickle('*****/hipfracture/python/datasets/final_dataset_preprocessed_v1.pkl')
df['filename_chest'] = df['filename_chest'].apply(lambda x: str(x)+'.png')
df['filename_hip'] = df['filename_hip'].apply(lambda x: str(x)+'.png')
df['PP_overl30d'] = df['PP_overl30d'].astype(int).astype(str)

df.set_index('PATIENTNR', inplace = True)

image_train = df.loc[X_train.index,['filename_hip','filename_chest','PP_overl30d']] 
image_val = df.loc[X_val.index,['filename_hip','filename_chest','PP_overl30d']] 
image_test = df.loc[X_test.index,['filename_hip','filename_chest','PP_overl30d']]

X_train = X_train.to_numpy()
X_val = X_val.to_numpy()
X_test = X_test.to_numpy()



if new_model:
        
    # callbacks to use during training
    early_stop = EarlyStopping(monitor='val_auc', mode = 'max', patience=10, verbose=1, min_delta=1e-4, restore_best_weights = True)
    reduce_lr = ReduceLROnPlateau(monitor='val_auc', mode = 'max', factor=0.05, patience=5, verbose=1, min_delta=1e-4)
    csv_logger = CSVLogger('*****/hipfracture/csv_logs/log_' + experiment_name + ".csv", append=True, separator=';')
    checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True, monitor='val_auc', mode='max')
    
    
    
    
    base_model_chest = xception.Xception(input_shape=(image_dim_chest, image_dim_chest, 3),
                                          weights=weights,
                                          include_top=False,
                                          pooling='avg')
    
    
    base_model_hip = resnet.ResNet152(input_shape=(image_dim_hip, image_dim_hip, 3),
                                          weights=weights,
                                          include_top=False,
                                          pooling='avg')

    model = build_model(base_model_chest = base_model_chest, base_model_hip = base_model_hip)
    
        
    
    METRICS = [Precision(name='precision'),
               Recall(name='recall'),
               AUC(name='auc')]
    
        
    model.compile(loss='binary_crossentropy', optimizer=Adam(1e-3), metrics=METRICS)
    print("Experiment name {}".format(experiment_name))
    
    
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

    history = model.fit_generator(
        two_image_generator(generator = train_datagen,
                            dataframe = image_train,
                            directory_1 = DIR_chest,
                            directory_2 = DIR_hip,
                            batch_size = batch_size,
                            x_1_col = 'filename_chest',
                            x_2_col = 'filename_hip',
                            y_col = 'PP_overl30d',
                            class_mode = 'binary',
                            shuffle = False,
                            img_size1 = (image_dim_chest,image_dim_chest),
                            img_size2 = (image_dim_hip,image_dim_hip)),                       
        steps_per_epoch = int(np.ceil(len(image_train) /batch_size)),
        epochs=100,
        validation_data=two_image_generator(generator = test_datagen,
                                            dataframe = image_val,
                                            directory_1 = DIR_chest,
                                            directory_2 = DIR_hip,
                                            batch_size = batch_size,
                                            x_1_col = 'filename_chest',
                                            x_2_col = 'filename_hip',
                                            y_col = 'PP_overl30d',
                                            class_mode = 'binary',
                                            shuffle = False,
                                            img_size1 = (image_dim_chest,image_dim_chest),
                                            img_size2 = (image_dim_hip,image_dim_hip)),
        validation_steps = int(np.ceil(len(image_val) /batch_size)),
        callbacks=[early_stop, reduce_lr, csv_logger, checkpointer])
    
    y_predict_scores = model.predict(two_image_generator(generator = test_datagen,
                                                    dataframe = image_val,
                                                    directory_1 = DIR_chest,
                                                    directory_2 = DIR_hip,
                                                    batch_size = batch_size,
                                                    x_1_col = 'filename_chest',
                                                    x_2_col = 'filename_hip',
                                                    y_col = 'PP_overl30d',
                                                    class_mode = 'binary',
                                                    shuffle = False, 
                                                    img_size1 = (image_dim_chest,image_dim_chest),
                                                    img_size2 = (image_dim_hip,image_dim_hip)),
                                 verbose = 1, steps = int(math.ceil(len(image_val) /batch_size)))

    y_predict = y_predict_scores >= 0.5
    
    y_true = y_val.astype(int)
        
    # saving validation results
    save_roc_curve(experiment_name = experiment_name, y_true = y_true, y_predict = y_predict )
    
    save_results(file_name_to_record_results,
                 y_true = y_true, y_predict = y_predict, 
                 y_predict_scores = y_predict_scores,
                 experiment_name = experiment_name,
                 experiment_notes = experiment_notes,
                 classifier = classifier)
    
    y_predict_scores = model.predict(two_image_generator(generator = test_datagen,
                                                        dataframe = image_test,
                                                        directory_1 = DIR_chest,
                                                        directory_2 = DIR_hip,
                                                        batch_size = 1,
                                                        x_1_col = 'filename_chest',
                                                        x_2_col = 'filename_hip',
                                                        y_col = 'PP_overl30d',
                                                        class_mode = 'binary',
                                                        shuffle = False, 
                                                        img_size1 = (image_dim_chest,image_dim_chest),
                                                        img_size2 = (image_dim_hip,image_dim_hip)),
                                     verbose = 1, steps = int(len(image_test)))
    
    y_predict = y_predict_scores >= 0.5
    
    y_true = y_test.astype(int)
    
    # saving validation results
    save_roc_curve(experiment_name = experiment_name, y_true = y_true, y_predict = y_predict )
    
    save_results(file_name_to_record_results,
                 y_true = y_true, y_predict = y_predict, 
                 y_predict_scores = y_predict_scores,
                 experiment_name = experiment_name,
                 experiment_notes = experiment_notes+'applied on test set',
                 classifier = classifier)

else:
    model = load_model(filepath)    
    test_datagen = ImageDataGenerator()
    y_predict_scores = model.predict(two_image_generator(generator = test_datagen,
                                                        dataframe = image_test,
                                                        directory_1 = DIR_chest,
                                                        directory_2 = DIR_hip,
                                                        batch_size = 1,
                                                        x_1_col = 'filename_chest',
                                                        x_2_col = 'filename_hip',
                                                        y_col = 'PP_overl30d',
                                                        class_mode = 'binary',
                                                        shuffle = False, 
                                                        img_size1 = (image_dim_chest,image_dim_chest),
                                                        img_size2 = (image_dim_hip,image_dim_hip)),
                                     verbose = 1, steps = int(len(image_test)))
    
    y_predict = y_predict_scores >= 0.5
    
    y_true = y_test.astype(int)
    
    # saving validation results
    save_roc_curve(experiment_name = experiment_name, y_true = y_true, y_predict = y_predict )
    
    save_results(file_name_to_record_results,
                 y_true = y_true, y_predict = y_predict, 
                 y_predict_scores = y_predict_scores,
                 experiment_name = experiment_name,
                 experiment_notes = experiment_notes+'applied on test set',
                 classifier = classifier)