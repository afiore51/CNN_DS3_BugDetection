import path
import Utils.constant as c
import Utils.functions as func
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import javalang
import pandas as pd
import os
from os.path import join
import glob
import numpy as np
import pprint
import pickle
import random
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report, f1_score
from imblearn.metrics import geometric_mean_score
import seaborn as sns
from sklearn.metrics import recall_score
from imblearn.metrics import specificity_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Input
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Convolution1D, MaxPooling1D
from tensorflow.keras.layers import BatchNormalization, Dropout, concatenate
from tensorflow.keras import Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
import plotly.graph_objects as go
import plotly.express as px
import re
from pprint import pprint
import time

def save_dict(obj, name):
    with open('DATA/dictionarys/' + name + '.pkl', 'wb') as ff:
        pickle.dump(obj, ff, pickle.HIGHEST_PROTOCOL)


def load_dict(name):
    with open('DATA/dictionarys/' + name + '.pkl', 'rb') as ff:
        return pickle.load(ff)


def convert_words(previous, current, verbose=False):
    global embeddedpath
    max_length = 0
    words = np.array([], dtype='object')
    for index, row in previous.iterrows():
        previous_file = embeddedpath + row.pathfolder[7:]
        # print(previous_file)
        with open(previous_file, 'rb') as f:
            m = 0
            for line in f:
                m += 1
                t = line.split()[0].decode('utf-8')

                words = np.append(words, t)

        if m > max_length:
            max_length = m
            max_file = previous_file

    for index, row in current.iterrows():
        current_file = embeddedpath + row.pathfolder[7:]
        # print(previous_file)
        with open(current_file, 'rb') as f:
            m = 0
            for line in f:
                m += 1
                t = line.split()[0].decode('utf-8')

                words = np.append(words, t)

        if m > max_length:
            max_length = m
            max_file = current_file

    # k = words
    if verbose:
        print(len(words))
        words = np.unique(words)
        print(words.shape)
        print('max lenght =' + str(max_length))

    print('convert_words finish')
    return max_length, words


def do_new_embedded(previous, current, dictionary, max_length):
    global embeddedpath
    for index, row in previous.iterrows():
        vectors = np.array([])
        previous_file = embeddedpath + row.pathfolder[7:]
        # print(row.pathfolder)
        Path(Path('DATA/new_embedded data' + row.pathfolder[7:] + '.embed').parent).mkdir(parents=True, exist_ok=True)
        Path(Path('DATA/new_embedded data' + row.pathfolder[7:] + '.embed')).touch()
        # print(previous_file)
        with open(previous_file, 'rb') as f:
            for line in f:
                t = line.split()[0].decode('utf-8')
                # print(t, '=>', dictionary[t])
                vectors = np.append(vectors, dictionary[t])
        vectors = vectors.astype('int32')
        if vectors.shape[0] < max_length:
            new_lenght = max_length - vectors.shape[0]
            vectors = np.append(vectors, np.zeros(new_lenght))
        np.savetxt('DATA/new_embedded data' + row.pathfolder[7:] + '.embed', vectors, delimiter=',', fmt='%i')

    for index, row in current.iterrows():
        vectors = np.array([])
        current_file = embeddedpath + row.pathfolder[7:]
        # print(row.pathfolder)
        Path(Path('DATA/new_embedded data' + row.pathfolder[7:] + '.embed').parent).mkdir(parents=True, exist_ok=True)
        # print(current_file)
        with open(current_file, 'rb') as f:
            for line in f:
                t = line.split()[0].decode('utf-8')
                # print(t, '=>', dictionary[t])
                vectors = np.append(vectors, dictionary[t])
        vectors = vectors.astype('int32')
        if vectors.shape[0] < max_length:
            new_lenght = max_length - vectors.shape[0]
            vectors = np.append(vectors, np.zeros(new_lenght))
        np.savetxt('DATA/new_embedded data' + row.pathfolder[7:] + '.embed', vectors, delimiter=',', fmt='%i')

    print('do_new_embedded finish')
    return


def generate_token(ant_token, version_tuple, classificatore, ds3=False, verbose=False, plot=False):
    global DS3
    DS3 = ds3
    if DS3:
        if verbose:
            print('______WITH DS3_____')
        typetest = 'DS3'


    else:
        if verbose:
            print('______WITHOUT DS3______')
        typetest = 'Without DS3'

    if verbose:
        # print(ant_token)
        print('Tuple versions ->', version_tuple)
    previous = ant_token[ant_token['pathfolder'].str.contains((version_tuple[0][:-4]))].reset_index().drop(
        columns=['index'])
    current = ant_token[ant_token['pathfolder'].str.contains((version_tuple[1][:-4]))].reset_index().drop(
        columns=['index'])

    print('Conver words starting...')
    max_length, words = convert_words(previous, current)

    tokens = np.array(random.sample(range(words.shape[0]), words.shape[0]))
    dictionary = dict(zip(words, tokens))

    # save_dict(dictionary, f'{version_tuple[0]} {version_tuple[1]}')

    print('Create new embedded files starting...')
    do_new_embedded(previous, current, dictionary, max_length)

    print('Create Dataset For CNN starting...')
    train_X, train_y, test_X, test_y = create_trainset_forCNN(version_tuple, previous, current, verbose)

    print('Create the CNN Model')
    model = create_model(train_X, tokens)

    print('Generate Sematic Features')
    predicted_train, predicted_test = generate_new_features(model, train_X, train_y, test_X, test_y)

    print('Creating Dataset for Regressor')
    ntrain_X, ntrain_y, ntest_X, ntest_y, resultcsv = create_dataset_forClass(previous, current, version_tuple,
                                                                              predicted_train, predicted_test, verbose)

    print('Starting Regression...')
    if classificatore == 'LogisticRegression':
        y_predicted = run_logisticRegression(ntrain_X, ntrain_y, ntest_X, ntest_y, verbose=verbose, plot=plot)
    if classificatore == 'RandomForest':
        y_predicted = run_random_forest(ntrain_X, ntrain_y, ntest_X, ntest_y, verbose = False, plot = False)

    print('Finished')



    resultcsv["bug predicted"] = y_predicted

    # pd.DataFrame(y_predicted).to_csv(r'Prediction for'+d.project[0] +f'versions {current.version[1]}.csv')
    print('Saving the results...')
    version_current = re.search("(\d+)\.(\d+)", version_tuple[1]).group()
    resultcsv.to_csv(
        f'.\Results\Prediction for {version_tuple[0][:version_tuple[0].index("-")]} version {version_current} CNN {classificatore} {typetest}.csv',
        index=False)
    return


def create_trainset_forCNN(version_tuple, previous, current, verbose=False):
    global mappeddataset
    ant_csv = glob.glob(mappeddataset + '/*.csv')

    ant_token = []
    for i in ant_csv:
        tmp = pd.read_csv(i, names=['pathfolder', 'label'])
        ant_token.append(tmp)
    ant_token = pd.concat(ant_token)
    ant_token_sorted = ant_token.sort_values(by='pathfolder')
    # ant_token =ant_token.reset_index().drop(['index'], axis= 1)
    ant_token_sorted = ant_token_sorted.reset_index().drop(['index'], axis=1)
    previous = ant_token_sorted[ant_token_sorted['pathfolder'].str.contains(version_tuple[0][:-4])].reset_index()
    current = ant_token_sorted[ant_token_sorted['pathfolder'].str.contains(version_tuple[1][:-4])].reset_index().drop(
        columns=['index'])

    vectors = []
    for index, row in previous.iterrows():
        previous_file = 'DATA/new_embedded data' + row.pathfolder[7:]
        # print(previous_file)
        with open(previous_file, 'rb') as f:
            l = []
            for line in f:
                l.append(int(line.split()[0]))

        vectors.append(l)

    train_X = np.array(vectors)
    train_y = previous['label'].to_numpy()

    if verbose:
        print('train_X shape ->', train_X.shape)
    vectors = []
    for index, row in current.iterrows():
        previous_file = 'DATA/new_embedded data' + row.pathfolder[7:]
        # print(previous_file)
        with open(previous_file, 'rb') as f:
            l = []
            for line in f:
                l.append(int(line.split()[0]))

        vectors.append(l)

    test_X = np.array(vectors)
    test_y = current['label'].to_numpy()

    if verbose:
        print('After create CNN SET train_X shape ->', train_X.shape)

    return train_X, train_y, test_X, test_y


def create_model(train_X, tokens, n_filters=15, filter_size=5):
    model = models.Sequential()
    model.add(layers.Embedding(tokens.shape[0], 32, input_length=train_X.shape[1],
                               embeddings_regularizer=tf.keras.regularizers.l2(l=0.0001)))
    model.add(
        Convolution1D(
            filters=32,
            strides=2,
            kernel_size=filter_size,
            kernel_initializer="he_normal",

        ))
    model.add(Activation('relu'))
    model.add(MaxPooling1D())
    model.add(
        Convolution1D(
            filters=32,
            strides=2,
            kernel_size=filter_size,
            kernel_initializer="he_normal",

        ))
    model.add(Activation('relu'))
    model.add(MaxPooling1D())
    model.add(
        Convolution1D(
            filters=64,
            strides=2,
            kernel_size=filter_size,
            kernel_initializer="he_normal",

        ))
    model.add(Activation('relu'))
    model.add(MaxPooling1D())
    model.add(layers.Flatten())
    model.add(Dense(256, kernel_regularizer=tf.keras.regularizers.l2(l=0.0001), kernel_initializer="he_normal"))
    model.add(Dropout(.50))
    model.add(Dense(256, kernel_regularizer=tf.keras.regularizers.l2(l=0.0001), kernel_initializer="he_normal"))
    model.add(Dropout(.50))
    model.add(Dense(256, kernel_regularizer=tf.keras.regularizers.l2(l=0.0001), kernel_initializer="he_normal"))
    model.add(Dropout(.50))
    model.add(Dense(256, kernel_regularizer=tf.keras.regularizers.l2(l=0.0001), kernel_initializer="glorot_normal"))

    # model.summary()

    return model


def generate_new_features(model, train_X, train_y, test_X, test_y):
    scaler = StandardScaler()
    scaler.fit(train_X)
    X_train = scaler.transform(train_X)
    y_train = train_y
    X_test = scaler.transform(test_X)
    y_test = test_y

    callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', min_delta=0.0001, patience=10)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0015), loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(train_X, train_y, epochs=20, callbacks=[callback], batch_size=256, verbose=0)

    predicted_train = model.predict(train_X)
    predicted_test = model.predict(test_X)

    return predicted_train, predicted_test


def create_dataset_forClass(previous, current, version_tuple, predicted_train, predicted_test, verbose=False):
    global DS3
    global classicdataset
    features = ['wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom',
                'ca', 'ce', 'npm', 'lcom3', 'loc', 'dam', 'moa', 'mfa', 'cam', 'ic',
                'cbm', 'amc', 'max_cc', 'avg_cc']
    ant_dataframe = []

    for filename in os.listdir(classicdataset):
        tmp = pd.read_csv(join(classicdataset, filename))
        ant_dataframe.append(tmp)
    ant_dataframe = pd.concat(ant_dataframe)

    previous_old = ant_dataframe.loc[
        ant_dataframe.version == float(re.search("(\d+)\.(\d+)", version_tuple[0]).group())].reset_index().drop(
        columns=['index'])
    current_old = ant_dataframe.loc[
        ant_dataframe.version == float(re.search("(\d+)\.(\d+)", version_tuple[1]).group())].reset_index().drop(
        columns=['index'])
    previous_old = previous_old.sort_values(by=['name']).reset_index().drop(columns=['index'])
    current_old = current_old.sort_values(by=['name']).reset_index().drop(columns=['index'])

    if DS3:
        D = func.create_D(current_old, previous_old, features, 'c')
        idx = func.runDS3(D, reg=.5, verbose=False)
        previous_old = previous_old.iloc[idx].reset_index().drop(columns=['index'])

    check = []
    ntrain_X = []
    ntrain_y = []
    for index, i in previous_old.iterrows():
        s = i['name']
        # print(s)
        for indexj, j in previous.iterrows():
            sj = j.pathfolder.replace('/', '.')
            # print(indexj)
            if (s in sj):
                # print(indexj)
                check.append(s)
                #             print(i[features])
                #             print(predicted_train[indexj])
                ntrain_X.append(np.concatenate((i[features], predicted_train[indexj]), axis=0))
                ntrain_y.append(i['bug'])

                break

    check = []
    ntest_X = []
    ntest_y = []
    for index, i in current_old.iterrows():
        s = i['name']
        # print(s)
        for indexj, j in current.iterrows():
            sj = j.pathfolder.replace('/', '.')
            # print(indexj)
            if (s in sj):
                # print(indexj)
                check.append(s)
                #             print(i[features])
                #             print(predicted_train[indexj])
                ntest_X.append(np.concatenate((i[features], predicted_test[indexj]), axis=0))
                ntest_y.append(i['bug'])

                break

    ntrain_X = np.array(ntrain_X)
    ntrain_y = np.array(ntrain_y)

    ntest_X = np.array(ntest_X)
    ntest_y = np.array(ntest_y)

    # print(y_predicted)
    resultcsv = pd.DataFrame(index=np.arange(len(check)))
    # y_predicted = pd.DataFrame(y_predicted, columns=["bug predicted"])

    resultcsv.insert(0, 'project', current_old.iloc[0][0])
    resultcsv.insert(1, 'version', current_old.iloc[0][1])

    resultcsv['name'] = check

    return ntrain_X, ntrain_y, ntest_X, ntest_y, resultcsv


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy, predictions


def run_random_forest(train_X, train_y, test_X, test_y, verbose=False, plot=False):
    """
        This function trains and uses model.

        :param D: matrix whose non-zero rows corresponds to the representatives of the dataset.
        :param p: norm to be used to calculate regularization cost.
        :returns: regularization cost.
    """

    scaler = StandardScaler()
    scaler.fit(train_X)
    X_train = scaler.transform(train_X)
    y_train = train_y
    X_test = scaler.transform(test_X)
    y_test = test_y
    ##PREDICT##

    # Create the random grid
    parameters = {
        'n_estimators': [200, 300, 320, 330, 340],
        'max_depth': [8, 9],
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
    if verbose:
        print('random forest score:', clf.score(X_train, y_train))
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


def run_logisticRegression(train_X, train_y, test_X, test_y, verbose=False, plot=False):
    scaler = StandardScaler()
    scaler.fit(train_X)
    X_train = scaler.transform(train_X)
    y_train = train_y
    X_test = scaler.transform(test_X)
    y_test = test_y

    clf = LogisticRegression(solver='liblinear')
    clf = clf.fit(X_train, y_train)
    ##PREDICT##
    y_predicted = clf.predict(X_test)
    print(y_predicted)
    ##CALCULATE SCORE OF THE MODEL##
    score = clf.score(X_test, y_test)
    if True:
        print(f'- LogisticRegression score: {score}')
    # CONFUCIO MATRIX##
    cm = metrics.confusion_matrix(y_test, y_predicted)
    ##PLOT CONFUSION MATRIX##
    if plot:
        plt.figure(figsize=(9, 9))
        sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        # all_sample_title = 'Confusion matrix \n Accuracy Score: {0}\n {1} {2}'.format(score, previous.version[0], current.version[1])
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        # plt.title(all_sample_title, size=15);
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


def start_run(mappeddat,embeddedp,classicdata,classificatore, ds3=False, verbose=False, plot=False):
    global mappeddataset
    global embeddedpath
    global classicdataset
    start_time = time.time()

    mappeddataset = mappeddat
    embeddedpath = embeddedp
    classicdataset = classicdata
    ant_csv = glob.glob(mappeddataset + '/*.csv')

    if verbose:
        print(ant_csv)
    ant_token = []

    for i in ant_csv:
        tmp = pd.read_csv(i, names=['pathfolder', 'label'])
        ant_token.append(tmp)

    ant_token = pd.concat(ant_token)
    ant_token['pathfolder'] = ant_token['pathfolder'].apply(lambda x: x.replace('.embed', ''))
    versions = [i.split('\\')[-1] for i in ant_csv]
    version_tuple = [(x, y) for x, y in zip(versions[0::1], versions[1::1])]
    if verbose:
        print(version_tuple)

    for v in version_tuple:
        if verbose:
            print(v)
        generate_token(ant_token, v, classificatore, ds3=ds3, verbose=verbose, plot=plot)

    pprint('All Done')
    pprint("--- %s seconds ---" % (time.time() - start_time))

