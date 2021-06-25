import pandas as pd

file_path_to_save_results = '*****/hipfracture/python/results/results.csv'

version_number = '1000_0_0_without_imaging'

X_train = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/X_train_imputated_v'+version_number+'.pkl')
X_val = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/X_val_imputated_v'+version_number+'.pkl')
X_test = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/X_test_imputated_v'+version_number+'.pkl')
    
y_train = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/y_train_v'+version_number+'.pkl')
y_val = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/y_val_v'+version_number+'.pkl')
y_test = pd.read_pickle('*****/hipfracture/python/datasets/dataset_v'+version_number+'/y_test_v'+version_number+'.pkl')

# choosing variables for AHFS
vars_ahfs = ['AGE', 'Geslacht_M', 'Geslacht_V', 'kwetsbaar_op_delerium',
             'geheugen_probl', 'HB', 'CCI_score_dbc', 'F_CAN',
             'woonsit_anders bepaald', 'woonsit_verpleeghuis',
            'woonsit_verzorgingshuis', 'woonsit_zelfstandig',
            'woonsit_zelfstandig met hulp', 'ASA2',
            'pre_fracture_mobility_geen functionele mobiliteit (gebruikmakend van onderste extremiteit)',
            'pre_fracture_mobility_mobiel binnenshuis maar nooit naar buiten zonder hulp',
            'pre_fracture_mobility_mobiel buiten met 1 hulpmiddel',
            'pre_fracture_mobility_mobiel buiten met 2 hulpmiddelen of frame (bv. rollator)',
            'pre_fracture_mobility_mobiel zonder hulpmiddelen', 'Kwetsbaar_op_ondervoeding', 'SNAQ_core',
            'onbedoeld_afgevallen', 'Verminderde_eetlust', 'drink_of_sondevoeding','Katz_adl_score']

# merging train and val sets
X_train = pd.concat([X_train, X_val])
y_train = pd.concat([y_train, y_val])

X_train = X_train[vars_ahfs]
X_test = X_test[vars_ahfs]


# create classifier 
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(penalty='none', solver= 'sag')

clf.fit(X_train, y_train)

from result_saver import save_results

try:
    y_predict_scores = clf.predict_proba(X_test)[:,1]
except:
    y_predict_scores = clf.decision_function(X_test)
    
y_predict = clf.predict(X_test)
experiment_name = 'test_results_for_baseline_model'
classifier = clf.__class__.__name__

from result_saver import save_results

save_results(file_name=file_path_to_save_results,
             y_true = y_test,
             y_predict = y_predict,         
             y_predict_scores = y_predict_scores,
             experiment_name = experiment_name,
             experiment_notes = '',                         
             imputater = '',
             classifier = classifier,
             sampler = '')

