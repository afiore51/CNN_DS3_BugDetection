import Utils.constant as c
import Utils.functions as f

import pandas as pd
import os
from os.path import join
from itertools import combinations
import numpy as np
from scipy.spatial.distance import euclidean, hamming
import collections

from numpy import matlib

import proxmin
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import classification_report, f1_score
from imblearn.metrics import geometric_mean_score
from sklearn.preprocessing import StandardScaler

from joblib import dump, load
import plotly.graph_objects as go
from sklearn.metrics import recall_score
from imblearn.metrics import specificity_score
import plotly.express as px

needed = ['name', 'wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom',
          'ca', 'ce', 'npm', 'lcom3', 'loc', 'dam', 'moa', 'mfa', 'cam', 'ic',
          'cbm', 'amc', 'max_cc', 'avg_cc', 'bug']
features_withbug = ['wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom',
                    'ca', 'ce', 'npm', 'lcom3', 'loc', 'dam', 'moa', 'mfa', 'cam', 'ic',
                    'cbm', 'amc', 'max_cc', 'avg_cc', 'bug']
features = ['wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom',
            'ca', 'ce', 'npm', 'lcom3', 'loc', 'dam', 'moa', 'mfa', 'cam', 'ic',
            'cbm', 'amc', 'max_cc', 'avg_cc']
distances = ['e', 'm', 'c']


def start_run(dataset_csv, DS3, distance, plot, verbose=False):
    print('Reading CSV Folder')
    d = []
    for filename in os.listdir(dataset_csv):
        tmp = pd.read_csv(join(dataset_csv, filename))
        d.append(tmp)
    d = pd.concat(d)
    # print(d)
    versions = d.version.unique()
    names = d.name.unique()
    print('Creating Version_tuple')
    version_tuple = [(x, y) for x, y in zip(versions[0::1], versions[1::1])]
    if verbose:
        print(version_tuple)
    for v in version_tuple:
        print("__________________________")
        previous = d.loc[d.version == v[0]]
        current = d.loc[d.version == v[1]]
        previous = previous.sort_values('name').reset_index()
        current = current.sort_values('name').reset_index()
        previous = previous.drop(['index'], axis=1)
        current = current.drop(['index'], axis=1)
        modules_previous = previous.name.unique()
        modules_current = current.name.unique()
        if verbose:
            print("PREVIOUS Version:", previous.version[0])
            print("CURRENT Version:", current.version[0])
            print("DISTANCE Using:", distance)
        if DS3:
            if verbose:
                print('______WITH DS3_____')
            print('Creating Distance Matrix')
            typetest = 'DS3'
            D = f.create_D(current, previous, features, distance)
            idx = f.runDS3(D, reg=.5, verbose=False)
            print('Starting Logistc Regression')
            y_predicted = f.run_logisticRegression(previous, current, idx, ds3=True, verbose=False, plot=False)

        else:
            if verbose:
                print('______WITHOUT DS3______')
            typetest = 'Without DS3'
            print('Starting Logistc Regression')
            y_predicted = f.run_logisticRegression(previous, current, None, ds3=False, verbose=False, plot=False)
        print("__________________________")

        # print(y_predicted)
        resultcsv = pd.DataFrame()
        # y_predicted = pd.DataFrame(y_predicted, columns=["bug predicted"])
        resultcsv['projcet'] = current.project
        resultcsv['version'] = current.version
        resultcsv['name'] = current.name
        resultcsv["bug predicted"] = y_predicted

        # pd.DataFrame(y_predicted).to_csv(r'Prediction for'+d.project[0] +f'versions {current.version[1]}.csv')
        print('Saving the results')
        resultcsv.to_csv(
            f'.\Results\Prediction for {d.project.iloc[0]} version {current.version[1]} Logistic {typetest}.csv',
            index=False)