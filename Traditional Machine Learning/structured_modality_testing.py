import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
#############################################################################
# input area
random_seed = 21
print(random_seed)

path_to_load_model =''
# version number is required to determine which dataset to use
# see dataset version control file to know which version you need
version_number = '1000_0_0_without_imaging'
experiment_name = 'Testing optimizied structured random forest chest and structured'
experiment_notes = 'dataset_v'+version_number

new_model = True

# file path to save results
file_name_to_record_results = '*****/hipfracture/python/results/results.csv'

# define classifier

if new_model:
    clf = RandomForestClassifier(n_estimators = 50, criterion = 'gini',
            							 max_depth = 5, min_samples_split = 40,
            							 min_samples_leaf = 1,max_features = 'log2',
            							 max_leaf_nodes = 100, bootstrap = False,
            							 random_state = random_seed)
        
    # importing datasets
        
    X_train = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/X_train_imputated_v'+version_number+'.pkl')
    X_val = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/X_val_imputated_v'+version_number+'.pkl')
    X_test = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/X_test_imputated_v'+version_number+'.pkl')
    
    y_train = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/y_train_v'+version_number+'.pkl')
    y_val = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/y_val_v'+version_number+'.pkl')
    y_test = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/y_test_v'+version_number+'.pkl')
    
    X, y = pd.concat([X_train, X_val]) , pd.concat([y_train,y_val])

    # train classifier
    clf.fit(X, y)
    
else:
    clf = load(path_to_load_model) 



classifier = clf.__class__.__name__
# inference

y_predict = clf.predict(X_test)

# confidence or probability estimates
try:
    y_predict_scores = clf.predict_proba(X_test)[:,1]
except:
    y_predict_scores = clf.decision_function(X_test)
    


from roc_curve_generator import save_roc_curve
from result_saver import save_results

save_roc_curve(experiment_name = experiment_name, y_true = y_test, y_predict = y_predict_scores)
save_results(file_name=file_name_to_record_results,
             y_true = y_test,
             y_predict = y_predict,         
             y_predict_scores = y_predict_scores,
             experiment_name = experiment_name,
             experiment_notes = experiment_notes,                         
             imputater = '',
             classifier = classifier,
             sampler = '')





