

def save_results(file_name = 'results.csv', y_true = None, y_predict = None, y_predict_scores = None,
                 use_threshold = False, threshold = 0.5,
                 experiment_name = 'not defined', experiment_notes = 'not defined',
                 imputater = 'not defined', classifier = 'not defined', sampler = 'not defined'):
    """

    Parameters
    ----------
    file_name : TYPE, optional
        DESCRIPTION. The default is 'results.csv'.
    y_true : TYPE, optional
        DESCRIPTION. The default is None.
    y_predict : TYPE, optional
        DESCRIPTION. The default is None.
    y_predict_scores : TYPE, optional
        DESCRIPTION. The default is None.
    use_threshold : TYPE, optional
        DESCRIPTION. The default is False.
    threshold : TYPE, optional
        DESCRIPTION. The default is 0.5.
    experiment_name : TYPE, optional
        DESCRIPTION. The default is 'not defined'.
    experiment_notes : TYPE, optional
        DESCRIPTION. The default is 'not defined'.
    imputater : TYPE, optional
        DESCRIPTION. The default is 'not defined'.
    classifier : TYPE, optional
        DESCRIPTION. The default is 'not defined'.
    sampler : TYPE, optional
        DESCRIPTION. The default is 'not defined'.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    
    
    import csv
    import datetime    
    from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score,roc_auc_score
    from decimal import getcontext, Decimal
    
    # setting decimal precision
    getcontext().prec = 3
    
    # calculate auc before converting predictions to classes
    auc_= float(Decimal(roc_auc_score(y_true,y_predict_scores))/Decimal(1))
    # converting predictions to classes
    if use_threshold:
        if y_predict_scores.max()>1 or y_predict_scores.min()<0 :
            raise ValueError('y_predict_score cannot be >1 or <0')
        y_predict = y_predict_scores >= threshold
    
    # calculating performance metrics
    recall = float(Decimal(recall_score(y_true,y_predict, pos_label = 1))/Decimal(1))
    precision = float(Decimal(precision_score(y_true,y_predict))/Decimal(1))
    f1 = float(Decimal(f1_score(y_true,y_predict))/Decimal(1))
    accuracy = float(Decimal(accuracy_score(y_true,y_predict))/Decimal(1))
    specificity = float(Decimal(recall_score(y_true,y_predict, pos_label = 0))/Decimal(1))
    date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(' ------------------------------ New Experiment results ------------------------------')
    
    try:
        f = open(file_name)
        f.close()
        print(file_name, 'found.')
    except:
        with open(file_name, "a", newline='') as csv_file:
            print('File not found, creating new file and adding header.')
            writer = csv.writer(csv_file, delimiter=';')
            line = ['experiment_name', 'experiment_notes','classifier', 'sampler', 'imputater',
                    'recall', 'precision', 'specificity', 'f1', 'accuracy','auc', 'experiment_date_time']
            writer.writerow(line)
            
    finally:
        with open(file_name, "a", newline='') as csv_file:
            print('Appending results')
            writer = csv.writer(csv_file, delimiter=';')
            
            line = [experiment_name, experiment_notes,classifier, sampler,imputater,
                    recall, precision, specificity, f1, accuracy, auc_, date_time]
            writer.writerow(line)
            
            

    print('Experiment name: ', experiment_name) 
    print ('Experiment notes: ', experiment_notes)
    print('Experiment date', date_time) 
    print('Classifier: ', classifier) 
    print ('Imputater: ', imputater)
    print('Sampler: ', sampler)         
    print (' -- > Recall score : {}'.format(recall))
    print (' -- > Precision score : {}'.format(precision))
    print (' -- > Specificity score : {}'.format(specificity))
    print (' -- > F1 score : {}'.format(f1))
    print (' -- > Accuracy score : {}'.format(accuracy))
    print (' -- > Area under the roc curve (AUC) : {}'.format(auc_))
    
def save_cross_validation_results(filepath = None, classifier = None , params = None, mean_score = None, stdev_score = None, split_scores = None,
                              experiment_name = None, experiment_notes = None):
    import csv
    import datetime    
    from decimal import getcontext, Decimal
    
    # setting decimal precision
    getcontext().prec = 3
    
    # calculate auc before converting predictions to classes
    mean_score = float(Decimal((mean_score))/Decimal(1))
    stdev_score = float(Decimal((stdev_score))/Decimal(1))
    classifier_name = classifier.__class__.__name__
    date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(' ------------------------------ New Experiment results ------------------------------')
    
    try:
        f = open(filepath)
        f.close()
        print(filepath, 'found.')
    except:
        with open(filepath, "a", newline='') as csv_file:
            print('File not found, creating new file and adding header.')
            writer = csv.writer(csv_file, delimiter=';')
            line = ['experiment_name', 'experiment_notes','classifier', 'hyperparameters',
                    'mean_auc', 'stdev_auc', 'split_scores', 'experiment_date_time']
            writer.writerow(line)
            
    finally:
        with open(filepath, "a", newline='') as csv_file:
            print('Appending results')
            writer = csv.writer(csv_file, delimiter=';')
            
            line = [experiment_name, experiment_notes,classifier_name, params,
                    mean_score, stdev_score, date_time]
            writer.writerow(line)
    print('Experiment name: ', experiment_name) 
    print('Experiment notes: ', experiment_notes)
    print('Experiment date', date_time) 
    print('Classifier: ', classifier_name) 
    print ('hyperparameters: ', params)
    print ('\n\n -- > mean_auc : {}'.format(mean_score))
    print (' -- > stdev_auc : {}'.format(stdev_score))
    print (' -- > split_scores : {}'.format(split_scores))
    
def main():
    import numpy as np
    y_true = np.array([1]*100 + [0]*100)
    y_predict = np.array(np.random.rand(200))
    save_results(file_name = '../results/test.csv',
                     y_true = y_true,
                     y_predict = y_predict>0.5,
                     y_predict_scores = y_predict,
                     experiment_name = 'testing_get_save_results',
                     experiment_notes = 'this is a dummy experiment by main(tester) function')
    

if __name__ == '__main__':
    main()
    
    
