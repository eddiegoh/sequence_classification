# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 17:43:47 2018

@author: Eddie
"""

from sklearn.metrics import confusion_matrix as cm
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import numpy as np

result = pd.read_csv('predictions.csv')
result.columns
y_true = result.Actual
y_pred = result.Predictions
classes = []
for i in range(1, 28):
    classes.append(i)


def plot_confusion_matrix(c_mat, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        c_mat = c_mat.astype('float') / c_mat.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(c_mat)

    plt.imshow(c_mat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(27)
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = c_mat.max() / 2.
    for i, j in itertools.product(range(c_mat.shape[0]), range(c_mat.shape[1])):
        plt.text(j, i, format(c_mat[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if c_mat[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Compute confusion matrix
matrix = cm(y_true, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(figsize=(27, 27), edgecolor='black')
plot_confusion_matrix(matrix, classes=classes,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure(figsize=(27, 27), edgecolor='black')
plot_confusion_matrix(matrix, classes=classes, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

