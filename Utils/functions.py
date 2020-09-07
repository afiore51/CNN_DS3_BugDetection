from Utils import constant as c
import pandas as pd
import os
from os.path import join
from itertools import combinations
import scipy as sp
from scipy.spatial.distance import euclidean, hamming, cityblock
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from Utils.DS3 import DS3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import classification_report, f1_score
from imblearn.metrics import geometric_mean_score
from sklearn.preprocessing import StandardScaler
import proxmin
from pathlib import Path
import joblib as joblib
from sklearn.metrics import recall_score
from imblearn.metrics import specificity_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.graph_objects as go




from joblib import dump, load


def chi2_distance(A, B):
    # compute the chi-squared distance using above formula
    chi = 0.5 * np.nansum([((a - b) ** 2) / (a + b)
                           for (a, b) in zip(A, B)])
    return chi


def create_D(current, previous, features, type):
    D = np.empty([len(previous), len(current)])
    for i in range(len(previous)):
        p = previous.loc[i, features]
        # print(p)
        for j in range(len(current)):
            c = current.loc[j, features]
            # print(c)
            if type == 'e':
                D[int(i), int(j)] = euclidean(p, c)
            if type == 'h':
                D[int(i), int(j)] = hamming(p, c)
            if type == 'm':
                D[int(i), int(j)] = cityblock(p, c)
            if type == 'c':
                D[int(i), int(j)] = chi2_distance(p, c)

    return D


def runDS3(D, reg, verbose=False):
    """
            This function runs DS3.

            :param D: matrix whose non-zero rows corresponds to the representatives of the dataset.
            :param p: norm to be used to calculate regularization cost.
            :returns: regularization cost.
    """
    # initialize DS3 class with dis-similarity matrix and the regularization parameter.
    dis_matrix = D
    reg = 0.5
    DS = DS3(dis_matrix, reg)
    # run the ADMM(p=inf) algorithm.
    start = time.time()
    data_admm, num_of_rep_admm, obj_func_value_admm, obj_func_value_post_proc_admm, Z = \
        DS.ADMM(mu=10 ** -1, epsilon=10 ** -7, max_iter=3000, p=np.inf)
    end = time.time()
    rep_super_frames = data_admm

    # change the above indices into 0s and 1s for all indices.
    N = len(D)
    summary = np.zeros(N)
    for i in range(len(rep_super_frames)):
        summary[rep_super_frames[i]] = 1

    run_time = end - start
    obj_func_value = obj_func_value_admm
    idx = []
    for index, i in enumerate(summary):
        if i == 1:
            idx.append(index)

    idx = np.asarray(idx)
    if verbose:
        print('Object function value :', obj_func_value)
        print("Run Time :", run_time)
        print("Objective Function Value  :", obj_func_value)
        print("Summary :", summary)
        print("Index representative :", idx)

    return idx


def run_logisticRegression(previous, current, idx, ds3=False, verbose=False, plot=False):
    """
        This function trains and uses model.

        :param D: matrix whose non-zero rows corresponds to the representatives of the dataset.
        :param p: norm to be used to calculate regularization cost.
        :returns: regularization cost.
    """

    scaler = StandardScaler()
    if ds3:
        training_D3 = previous.iloc[idx].reset_index()
        scaler.fit(training_D3[c.features])
        X_train = scaler.transform(training_D3[c.features])
        y_train = training_D3['bug']
        X_test = scaler.transform(current[c.features])
        y_test = current['bug']
    else:
        train = previous
        scaler.fit(train[c.features])
        X_train = scaler.transform(train[c.features])
        y_train = train['bug']
        X_test = scaler.transform(current[c.features])
        y_test = current['bug']

    # filename = join(c.MODEL_DIR , 'digits_classifier.joblib.pkl')

    # if Path(filename).is_file():
    #   print('Carico il modello')
    #   clf = joblib.load(filename)
    # else:
    #    print('Creo il modello')

    clf = LogisticRegression(solver='liblinear')
    clf = clf.fit(X_train, y_train)
    ##PREDICT##
    y_predicted = clf.predict(X_test)
    # print(y_predicted)
    ##CALCULATE SCORE OF THE MODEL##
    score = clf.score(X_test, y_test)
    if True:
        print(f'- LogisticRegression score: {score}')
    # CONFUCIO MATRIX##
    cm = metrics.confusion_matrix(y_test, y_predicted)
    ##PLOT CONFUSION MATRIX##
    if plot:
        plt.figure(figsize=(9, 9))
        sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r');
        plt.ylabel('Actual label');
        plt.xlabel('Predicted label');
        all_sample_title = 'Confusion matrix \n Accuracy Score: {0}\n {1} {2}'.format(score, previous.version[0],
                                                                                      current.version[1])
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.title(all_sample_title, size=15);
    cmm = metrics.multilabel_confusion_matrix(y_test, y_predicted)
    if verbose:
        print("Confusion multiclass matrix :\n", cmm)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, y_predicted))
        print("Accuracy:", metrics.accuracy_score(y_test, y_predicted))

    ############################################################################
    ############################################################################
    #    'micro':
    #     Calculate metrics globally by counting the total true positives, false negatives and false positives.
    #
    #   'macro':
    #     Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
    ############################################################################
    ############################################################################

    gmean = geometric_mean_score(y_test, y_predicted, average='micro')
    # print(cm)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    # TPR = TP / (TP + FN)
    TPR = recall_score(y_test, y_predicted, average='micro')

    # Fall out or false positive rate
    # FPR = FP / (FP + TN)
    FPR = 1 - (specificity_score(y_test, y_predicted, average='micro'))

    # Balance
    balance = 1 - (np.sqrt((0 - FPR) ** 2 + (1 - TPR) ** 2) / np.sqrt(2))
    balance = np.average(balance)

    ##F MEASURE##
    fmeasure = f1_score(y_test, y_predicted, average='micro')

    if verbose:
        print('TPR :', TPR)
        print('FPR :', FPR)

    print('F-Measure : ', fmeasure)
    print('G-Mean :', gmean)
    print('Balance :', balance)

    '''filename = join(c.MODEL_DIR , 'digits_classifier.joblib.pkl')
    _ = joblib.dump(clf, filename, compress=9)
    '''
    return y_predicted


def run_SVC(previous, current, idx, ds3=False, verbose=False, plot=False):
    """
        This function trains and uses model.

        :param D: matrix whose non-zero rows corresponds to the representatives of the dataset.
        :param p: norm to be used to calculate regularization cost.
        :returns: regularization cost.
    """

    scaler = StandardScaler()
    if ds3:
        training_D3 = previous.iloc[idx].reset_index()
        scaler.fit(training_D3[c.features])
        X_train = scaler.transform(training_D3[c.features])
        y_train = training_D3['bug']
        X_test = scaler.transform(current[c.features])
        y_test = current['bug']
    else:
        train = previous
        scaler.fit(train[c.features])
        X_train = scaler.transform(train[c.features])
        y_train = train['bug']
        X_test = scaler.transform(current[c.features])
        y_test = current['bug']

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                         'C': [0.001, 0.01, 0.1, 1, 10]},
                        {'kernel': ['linear'], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'C': [0.001, 0.01, 0.1, 1, 10]}]

    print("# Tuning hyper-parameters for %s" % 'precision')
    print()
    clf = GridSearchCV(
        SVC(), tuned_parameters, scoring='%s_macro' % 'precision'
    )

    clf = clf.fit(X_train, y_train)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    #     for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    #         print("%0.3f (+/-%0.03f) for %r"
    #               % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_predicted = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_predicted))
    print()
    print()
    ##CALCULATE SCORE OF THE MODEL##
    if plot:
        fig = px.scatter(y_test)
        fig.add_trace(go.Scatter(x=list(range(y_test.shape[0])), y=y_predicted))
        fig.show()

    cm = metrics.confusion_matrix(y_test, y_predicted)
    gmean = geometric_mean_score(y_test, y_predicted, average='micro')
    # print(cm)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    # TPR = TP / (TP + FN)
    TPR = recall_score(y_test, y_predicted, average='micro')

    # Fall out or false positive rate
    # FPR = FP / (FP + TN)
    FPR = 1 - (specificity_score(y_test, y_predicted, average='micro'))

    # Balance
    balance = 1 - (np.sqrt((0 - FPR) ** 2 + (1 - TPR) ** 2) / np.sqrt(2))
    balance = np.average(balance)

    ##F MEASURE##
    fmeasure = f1_score(y_test, y_predicted, average='micro')
    print('F-Measure : ', fmeasure)
    print('G-Mean :', gmean)
    print('Balance :', balance)
    ############################################################################
    ############################################################################
    #    'micro':
    #     Calculate metrics globally by counting the total true positives, false negatives and false positives.
    #
    #   'macro':
    #     Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
    ############################################################################
    ############################################################################

    '''filename = join(c.MODEL_DIR , 'digits_classifier.joblib.pkl')
    _ = joblib.dump(clf, filename, compress=9)
    '''
    return y_predicted


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy, predictions


def run_RandomForest(previous, current, idx, ds3=False, verbose=False, plot=False):
    """
        This function trains and uses model.

        :param D: matrix whose non-zero rows corresponds to the representatives of the dataset.
        :param p: norm to be used to calculate regularization cost.
        :returns: regularization cost.
    """

    scaler = StandardScaler()
    if ds3:
        training_D3 = previous.iloc[idx].reset_index()
        scaler.fit(training_D3[c.features])
        X_train = scaler.transform(training_D3[c.features])
        y_train = training_D3['bug']
        X_test = scaler.transform(current[c.features])
        y_test = current['bug']
    else:
        train = previous
        scaler.fit(train[c.features])
        X_train = scaler.transform(train[c.features])
        y_train = train['bug']
        X_test = scaler.transform(current[c.features])
        y_test = current['bug']

    # Create the random grid
    parameters = {
        'n_estimators': [200, 300, 320, 330, 340],
        'max_depth': [8, 9, 10, 11, 12],
        'random_state': [0],
        'max_features': ['auto', 'sqrt', 'log2']
        # 'max_features': ['auto'],
        # 'criterion' :['gini']
    }

    # Random search of parameters,
    # search across 100 different combinations, and use all available cores
    clf = GridSearchCV(RandomForestClassifier(),
                       parameters, cv=10, n_jobs=-1)

    clf = clf.fit(X_train, y_train)
    print(clf.score(X_train, y_train))
    print(clf.best_params_)
    y_predicted = clf.predict(X_test)

    ##CALCULATE SCORE OF THE MODEL##
    if plot:
        fig = px.scatter(y_test)
        fig.add_trace(go.Scatter(x=list(range(y_test.shape[0])), y=y_predicted))
        fig.show()

    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print(classification_report(y_test, y_predicted))
    print()
    print()

    cm = metrics.confusion_matrix(y_test, y_predicted)
    gmean = geometric_mean_score(y_test, y_predicted, average='micro')
    # print(cm)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    # TPR = TP / (TP + FN)
    TPR = recall_score(y_test, y_predicted, average='micro')

    # Fall out or false positive rate
    # FPR = FP / (FP + TN)
    FPR = 1 - (specificity_score(y_test, y_predicted, average='micro'))

    # Balance
    balance = 1 - (np.sqrt((0 - FPR) ** 2 + (1 - TPR) ** 2) / np.sqrt(2))
    balance = np.average(balance)

    ##F MEASURE##
    fmeasure = f1_score(y_test, y_predicted, average='micro')
    print('F-Measure : ', fmeasure)
    print('G-Mean :', gmean)
    print('Balance :', balance)
    ############################################################################
    ############################################################################
    #    'micro':
    #     Calculate metrics globally by counting the total true positives, false negatives and false positives.
    #
    #   'macro':
    #     Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
    ############################################################################
    ############################################################################

    '''filename = join(c.MODEL_DIR , 'digits_classifier.joblib.pkl')
    _ = joblib.dump(clf, filename, compress=9)
    '''
    return y_predicted
