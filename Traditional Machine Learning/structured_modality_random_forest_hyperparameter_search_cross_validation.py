import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier



version_number = '1000_0_0_without_imaging'

file_path_to_save_results = 'hyper_parameter_search_results_v1000_1.pkl'

# retrieving prepared datasets

X_train = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/X_train_imputated_v'+version_number+'.pkl')
X_val = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/X_val_imputated_v'+version_number+'.pkl')

y_train = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/y_train_v'+version_number+'.pkl')
y_val = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/y_val_v'+version_number+'.pkl')

# concatenating train and validation set because we will do cross validation
X, y = pd.concat([X_train, X_val]) , pd.concat([y_train,y_val])
print(X.shape, y.shape)


from imblearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import StratifiedKFold

# define k for k fold cross validation
skf = StratifiedKFold(n_splits=5)

random_seed = 21

model = Pipeline([
        ('classification', RandomForestClassifier(random_state = random_seed))
    ])

# hyperparameters to validate

param_grid = {'classification__n_estimators':[10,20,30,40],
              'classification__max_depth':[10,12,15],
              'classification__min_samples_leaf':[1,2,3],
              'classification__min_samples_split':[40,50,60],
              'classification__max_features' : ['log2'],
              'classification__criterion':['gini'],
              'classification__max_leaf_nodes':[None, 50,100],
              'classification__bootstrap':[False,True]
              }

auc_scorer = make_scorer(score_func = roc_auc_score,
                         greater_is_better = True,
                         needs_proba = True)

clf = GridSearchCV(estimator = model, param_grid = param_grid, scoring = auc_scorer,
                   n_jobs = -1, cv = skf, verbose = 3, return_train_score = True, refit = False)


clf.fit(X, y)

df_results = pd.DataFrame(clf.cv_results_)

df_results.to_pickle(file_path_to_save_results)

