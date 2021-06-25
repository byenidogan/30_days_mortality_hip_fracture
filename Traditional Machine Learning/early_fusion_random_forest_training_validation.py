import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer

# this experiment type is validating model agnostic multimodal learning techniques
# validated random forest classifier used on the datasets which include image classifier outputs as well

# make it true if you want to include hip modality
include_hip = False


for run_nr in range(5):
    # changing seed value for randomization in experiments
    random_seed = ((21+run_nr*run_nr))
    print(random_seed)
    
    # version number is required to determine which dataset to use
    # see dataset version control file to know which version you need
    version_number = '1000_0_0_with_imaging_inner_layer'

    # change name and notes based on the experiment
    experiment_name = 'experiment_v1000_0_normal_validation strucutred and chest' + version_number
    
    experiment_notes = 'run number:'+ str(run_nr) + ', validating_datasets'
    
    file_name_to_record_results = '*****/hipfracture/python/results/results.csv'
    
    
    X_train = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/X_train_v'+version_number+'.pkl')
    X_val = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/X_val_v'+version_number+'.pkl')
    X_test = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/X_test_v'+version_number+'.pkl')
    
    
    if not include_hip:
        for i in range(8):
            X_train.pop('hip_finding_'+str(i))
            X_val.pop('hip_finding_'+str(i))
            X_test.pop('hip_finding_'+str(i))
            
    y_train = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/y_train_v'+version_number+'.pkl')
    y_val = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/y_val_v'+version_number+'.pkl')
    y_test = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/y_test_v'+version_number+'.pkl')
    
    # classifier imports
    
    from sklearn.ensemble import RandomForestClassifier
    
    from roc_curve_generator import save_roc_curve
    from result_saver import save_results
    
    
    # classifier training and prediction
    classifier = RandomForestClassifier(n_estimators = 50, criterion = 'gini',
                                         max_depth = 5, min_samples_split = 40,
                                         min_samples_leaf = 1,max_features = 'log2',
                                         max_leaf_nodes = 100, bootstrap = False,
                                         random_state = random_seed)
    
    classifier.fit(X_train, y_train)
    
    y_predict = classifier.predict(X_val)
    
    # confidence or probability estimates
    try:
        y_predict_scores = classifier.predict_proba(X_val)[:,1]
    except:
        y_predict_scores = classifier.decision_function(X_val)
    
    save_roc_curve(experiment_name = experiment_name, y_true = y_val, y_predict = y_predict_scores )
    save_results(file_name=file_name_to_record_results,
                 y_true = y_val,
                 y_predict = y_predict,         
                 y_predict_scores = y_predict_scores,
                 experiment_name = experiment_name,
                 experiment_notes = experiment_notes,                         
                 imputater = '',
                 classifier = '',
                 sampler = '')
    


