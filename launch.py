import numpy as np
import random as rd
from FileIO import load
from Components import *
from Visualisation import plotNumbers, plotTrainTestPerformance, plotWrongDigits, tsne_plots
import sys

# loading data
x_train, y_train, x_test, y_test = load(.5, plot=False)

# baseline model with random parameters based on lit (blog) research
# print(randomParameters())

# # print tSNE plots

# model, _ = train_cnn(x_train, y_train)
# acc,_,_,_, feature_vectors = eval_cnn(model, x_test, y_test, tSNE_list=True)
# print(acc)
# tsne_plots(x_test, y_test, feature_vectors)

# cross-validation

train, test, m, change, saveLocation, append = crossvalidationCNN(x_train, y_train, 10)
if not append:
   plotTrainTestPerformance(train, test, change, saveLocation, x_values=m)


# show division of data set in training vs testing
# data_analysis(x_train, y_train, x_test, y_test)

# linear regression
'''
print(linear_regression(x_train,y_train, x_test, y_test))
trainingError = list()
testingError = list()

for m in range(1, 200):
    pcaTrain, pcaTest, percentageExplainedPC = pca(x_train, x_test, nComponents=m, plot=False)
    results = linear_regression(pcaTrain, y_train, pcaTest, y_test)
    trainingError.append(results[0])
    testingError.append(results[1])
plotTrainTestPerformance(trainingError, testingError, 'Principal Components')
'''


# Train and test several models for average testing accuracy

# test_model(x_train,y_train, x_test, y_test)

