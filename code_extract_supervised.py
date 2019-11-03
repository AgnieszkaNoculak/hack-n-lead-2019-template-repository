#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 17:39:50 2019

@author: elenalabzina
"""


#THIS IS NOT ALL OF THE USED CODE BUT JUST TO GIVE YOU SOME IDEA

import sys

import eli5
from eli5.sklearn import PermutationImportance

import scipy
import numpy
import matplotlib
import pandas
import sklearn
from IPython.display import display, HTML
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import permutation_test_score
import numpy as np
import pandas as pd



def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

# import data 

df = pd.read_csv('/Users/elenalabzina/Dropbox/CS_challenge/data/large.csv')


df_with_dummies = pd.get_dummies(df, prefix='nat_', columns=['nationality'])

df_with_dummies.to_csv('/Users/elenalabzina/Dropbox/CS_challenge/data/not_outliers_with_dummies.csv')

d = df_with_dummies
d1= pd.read_csv('/Users/elenalabzina/Dropbox/CS_challenge/data/jeopardy.csv')

d = d.merge(d1, on='cif',how='inner')


#training without rebalancing

validation_size = 0.3
seed=10

count_class_0, count_class_1 = d.suspicious.value_counts()

#count_class_0
#Out[49]: 982517

#count_class_1
#Out[50]: 17109

# Divide by class
df_class_0 = d[d['suspicious'] == 0]
df_class_1 = d[d['suspicious'] == 1]

df_class_0_under = df_class_0.sample(count_class_1)
df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)
d = df_test_under

outcome = 'suspicious'

features = [ 'category', 'turnover', 'transaction_count', 'io_ratio', 'age',
       'is_pep', 'inactive_days_average', 'inactive_days_max', 'n_of_accounts',
       'distinct_counterparties', 'channel_risk', 'atm_withdrawal',
       'atm_deposit', 'nat__24', 'nat__32', 'nat__33', 'nat__47', 'nat__82',
       'nat__90', 'nat__113', 'nat__123', 'nat__125', 'nat__133', 'nat__138',
       'nat__145', 'nat__151', 'nat__166', 'nat__186', 'nat__187']
X = d[features]
Y = d[outcome]

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size)

from imblearn.combine import SMOTETomek
smote_tomek = SMOTETomek(random_state=0)
X_resampled, Y_resampled = smote_tomek.fit_resample(X_train, Y_train)

GBC = GradientBoostingClassifier(n_estimators=100, learning_rate=1)
y_pred = GBC.fit(X_resampled, Y_resampled).predict(X_validation)

# Plot non-normalized confusion matrix
plot_confusion_matrix(Y_validation, y_pred, classes=Y.unique(),
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(Y_validation, y_pred, classes=Y.unique(), normalize=True,
                      title='Normalized confusion matrix')


LR = LogisticRegression(solver='liblinear', multi_class='ovr')
y_pred = GBC.fit(X_resampled, Y_resampled).predict(X_validation)

# Plot non-normalized confusion matrix
plot_confusion_matrix(Y_validation, y_pred, classes=Y.unique(),
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(Y_validation, y_pred, classes=Y.unique(), normalize=True,
                      title='Normalized confusion matrix')

perm = PermutationImportance(GBC, cv = None, refit = True, n_iter = 50).fit(X_resampled, Y_resampled)
feature_imp2 = pd.DataFrame(list(zip(X.columns,perm.feature_importances_)),columns=['feature','importance'])
feature_imp2 = feature_imp2.sort_values(by=['importance'],ascending=False)
feature_imp2.reset_index()
print(feature_imp2)

pd.concat([pd.DataFrame(features),pd.DataFrame(LR.coef_).T ], axis=1)











