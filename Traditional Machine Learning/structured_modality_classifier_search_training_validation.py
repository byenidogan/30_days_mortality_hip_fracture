import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate

# classifier imports
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import StratifiedKFold
import csv
random_seed = 21

experiment_name = 'structured_modality_classifier_crossvalidated_search_1000_0_0_without_imaging'
version_number = '1000_0_0_without_imaging'

filepath = 'results_structured_modality_classifier_crossvalidated_search.csv'

imputater_name = 'k_neighbors'

# data import

X_train = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/X_train_imputated_v'+version_number+'.pkl')
X_val = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/X_val_imputated_v'+version_number+'.pkl')
X_test = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/X_test_imputated_v'+version_number+'.pkl')

y_train = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/y_train_v'+version_number+'.pkl')
y_val = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/y_val_v'+version_number+'.pkl')
y_test = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/y_test_v'+version_number+'.pkl')

X, y = pd.concat([X_train, X_val]) , pd.concat([y_train,y_val])

print(X.shape, y.shape)

# classifiers

logistic_regression = LogisticRegression(solver='liblinear',max_iter = 10000, random_state=random_seed)
random_forest_classifier = RandomForestClassifier( n_estimators = 100, random_state=random_seed)
adaboost_classifier = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(), n_estimators=100, random_state=random_seed)
xgb_classifier = XGBClassifier(random_state = random_seed)

classifiers = [logistic_regression,random_forest_classifier, 
              adaboost_classifier,xgb_classifier]

skf = StratifiedKFold(n_splits=5)


auc_scorer = make_scorer(score_func = roc_auc_score,
                         greater_is_better = True,
                         needs_proba = True)

for model in classifiers:

    scores = cross_validate(model, X, y, scoring=auc_scorer, return_train_score = True)

    classifier_name = model.__class__.__name__
    
    mean_train = scores['train_score'].mean()
    std_train = scores['train_score'].std()
    
    mean_val = scores['test_score'].mean()
    std_val = scores['test_score'].std()
    
    
    try:
       f = open(filepath)
       f.close()
       print(filepath, 'found.')
    except:
        with open(filepath, "a", newline='') as csv_file:
            print('File not found, creating new file and adding header.')
            writer = csv.writer(csv_file, delimiter=';')
            line = ['experiment_name', 'classifier', 'mean_val_score','std_val_score', 'mean_train_score',
                    'std_train_score']
            writer.writerow(line)
            
    finally:
        with open(filepath, "a", newline='') as csv_file:
            print('Appending results')
            writer = csv.writer(csv_file, delimiter=';')
            
            line = [experiment_name, classifier_name, mean_val,
                    std_val, mean_train, std_train]
            writer.writerow(line)
            
# section for linear SVC
# it needs seperate section because it does not have predict_proba function
        
linear_SVC = LinearSVC(loss='hinge', random_state = random_seed)
classifier_name = linear_SVC.__class__.__name__
X,y = X.to_numpy(), y.to_numpy()
skf.get_n_splits(X, y)
auc_scores_test = []
auc_scores_train = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    linear_SVC.fit(X_train,y_train)
    
    prediction_scores_test = linear_SVC.decision_function(X_test)
    prediction_scores_train = linear_SVC.decision_function(X_train)
    
    auc_scores_test.append(roc_auc_score(y_test,prediction_scores_test))
    auc_scores_train.append(roc_auc_score(y_train,prediction_scores_train))
    
    
auc_scores_test = np.array(auc_scores_test)
auc_scores_train = np.array(auc_scores_train)

mean_train = auc_scores_train.mean()
std_train = auc_scores_train.std()

mean_val = auc_scores_test.mean()
std_val = auc_scores_test.std()
try:
   f = open(filepath)
   f.close()
   print(filepath, 'found.')
except:
    with open(filepath, "a", newline='') as csv_file:
        print('File not found, creating new file and adding header.')
        writer = csv.writer(csv_file, delimiter=';')
        line = ['experiment_name', 'classifier', 'mean_val_score','std_val_score', 'mean_train_score',
                'std_train_score']
        writer.writerow(line)
        
finally:
    with open(filepath, "a", newline='') as csv_file:
        print('Appending results')
        writer = csv.writer(csv_file, delimiter=';')
        
        line = [experiment_name, classifier_name, mean_val,
                std_val, mean_train, std_train]
        writer.writerow(line)
   
