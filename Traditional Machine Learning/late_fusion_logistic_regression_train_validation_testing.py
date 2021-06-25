import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
#############################################################################
# input area


path_to_load_model =''
# version number is required to determine which dataset to use
# see dataset version control file to know which version you need
version_number = '1000_0_0_with_imaging_output_layer'
experiment_name = 'Experiment late fusion - strucutred and chest'
experiment_notes = 'dataset_v'+version_number

new_model = True

validation = False

include_hip = False

# file path to save results
file_name_to_record_results = '*****/hipfracture/python/results/results.csv'

if validation:
    
    for run_nr in range(5):
        # changing seed value for randomization in experiments
        random_seed = ((21+run_nr*run_nr))
        print(random_seed)
        # define classifier
        
        
        if new_model:
            clf = RandomForestClassifier(n_estimators = 50, criterion = 'gini',
            							 max_depth = 5, min_samples_split = 40,
            							 min_samples_leaf = 1,max_features = 'log2',
            							 max_leaf_nodes = 100, bootstrap = False,
            							 random_state = random_seed)
                
            # importing datasets
                
            X_train = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/X_train_v'+version_number+'.pkl')
            X_train_chest = X_train.pop('chest_finding_0')
            X_train_hip = X_train.pop('hip_finding_0')
            X_val = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/X_val_v'+version_number+'.pkl')
            X_val_chest = X_val.pop('chest_finding_0')
            X_val_hip = X_val.pop('hip_finding_0')
            X_test = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/X_test_v'+version_number+'.pkl')
            X_test_chest = X_test.pop('chest_finding_0')
            X_test_hip = X_test.pop('hip_finding_0')
            
            y_train = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/y_train_v'+version_number+'.pkl')
            y_val = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/y_val_v'+version_number+'.pkl')
            y_test = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/y_test_v'+version_number+'.pkl')
            
            image_features = ['chest_finding_0', 'hip_finding_0']
            
            # train classifier
            clf.fit(X_train, y_train)
            
        else:
            clf = load(path_to_load_model) 
        
        
        
        
        y_score_train = clf.predict_proba(X_train)[:,1]
        
        y_score_val = clf.predict_proba(X_val)[:,1]
        
        if include_hip:
            X_train = pd.DataFrame({'RF_Structured_decision_output':y_score_train,
                                    'chest_decision_output':X_train_chest,
                                    'hip_decision_output':X_train_hip})
            
            X_val = pd.DataFrame({'RF_Structured_decision_output':y_score_val,
                                    'chest_decision_output':X_val_chest,
                                    'hip_decision_output':X_val_hip})
            
        else:
            X_train = pd.DataFrame({'RF_Structured_decision_output':y_score_train,
                                    'chest_decision_output':X_train_chest})
            
            X_val = pd.DataFrame({'RF_Structured_decision_output':y_score_val,
                                    'chest_decision_output':X_val_chest})
        
        
        
        from sklearn.linear_model import LogisticRegression
        
        lr = LogisticRegression(random_state=random_seed, max_iter=10000)
        
        
        lr.fit(X_train, y_train)
        
        y_predict = lr.predict(X_val)
        y_predict_scores = lr.predict_proba(X_val)[:,1]
        
        # from roc_curve_generator import save_roc_curve
        from result_saver import save_results
        
        # save_roc_curve(experiment_name = experiment_name, y_true = y_test, y_predict = y_predict_scores)
        save_results(file_name=file_name_to_record_results,
                      y_true = y_val,
                      y_predict = y_predict,         
                      y_predict_scores = y_predict_scores,
                      experiment_name = experiment_name,
                      experiment_notes = 'late fusion - normal validation',                         
                      imputater = '',
                      classifier = lr.__class__.__name__,
                      sampler = '')
        
    
else:
     
    if new_model:
        clf = RandomForestClassifier(n_estimators = 50, criterion = 'gini',
        							 max_depth = 5, min_samples_split = 40,
        							 min_samples_leaf = 1,max_features = 'log2',
        							 max_leaf_nodes = 100, bootstrap = False,
        							 random_state = random_seed)  


    
        # importing datasets
            
        
        X_train = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/X_train_v'+version_number+'.pkl')
        X_train_chest = X_train.pop('chest_finding_0')
        X_train_hip = X_train.pop('hip_finding_0')
        X_val = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/X_val_v'+version_number+'.pkl')
        X_val_chest = X_val.pop('chest_finding_0')
        X_val_hip = X_val.pop('hip_finding_0')
        X_test = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/X_test_v'+version_number+'.pkl')
        X_test_chest = X_test.pop('chest_finding_0')
        X_test_hip = X_test.pop('hip_finding_0')
        
        y_train = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/y_train_v'+version_number+'.pkl')
        y_val = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/y_val_v'+version_number+'.pkl')
        y_test = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/y_test_v'+version_number+'.pkl')
        
        image_features = ['chest_finding_0', 'hip_finding_0']
        
        X, y = pd.concat([X_train, X_val]) , pd.concat([y_train,y_val])
        # train classifier
        clf.fit(X, y)
    else:
        clf = load(path_to_load_model) 
    
        
    
    X_train_chest = pd.concat([X_train_chest,X_val_chest])
    X_train_hip = pd.concat([X_train_hip, X_val_hip])
    y_train = pd.concat([y_train, y_val])
    y_score_train = clf.predict_proba(X)[:,1]
    
    y_score_test = clf.predict_proba(X_test)[:,1]
    
    if include_hip:
            X_train = pd.DataFrame({'RF_Structured_decision_output':y_score_train,
                                    'chest_decision_output':X_train_chest,
                                    'hip_decision_output':X_train_hip})
            
            X_test = pd.DataFrame({'RF_Structured_decision_output':y_score_test,
                                    'chest_decision_output':X_test_chest,
                                    'hip_decision_output':X_test_hip})
            
    else:
        X_train = pd.DataFrame({'RF_Structured_decision_output':y_score_train,
                                'chest_decision_output':X_train_chest})
        
        X_test = pd.DataFrame({'RF_Structured_decision_output':y_score_test,
                                    'chest_decision_output':X_test_chest})
    
    
    from sklearn.linear_model import LogisticRegression
    
    lr = LogisticRegression(random_state=random_seed, max_iter=10000)
    
    
    lr.fit(X_train, y_train)
    
    y_predict = lr.predict(X_test)
    y_predict_scores = lr.predict_proba(X_test)[:,1]
    
    # from roc_curve_generator import save_roc_curve
    from result_saver import save_results
    
    # save_roc_curve(experiment_name = experiment_name, y_true = y_test, y_predict = y_predict_scores)
    save_results(file_name=file_name_to_record_results,
                  y_true = y_test,
                  y_predict = y_predict,         
                  y_predict_scores = y_predict_scores,
                  experiment_name = experiment_name,
                  experiment_notes = 'late fusion - test results',                         
                  imputater = '',
                  classifier = lr.__class__.__name__,
                  sampler = '')
    