import numpy as np
from fileIO import load
from Components import pca, linear_regression, vectors_to_matrices
from CNN import cnn
from Visualisation import plotNumbers, plotTrainTestPerformance


x_train, y_train, x_test, y_test = load(.8)
print(linear_regression(x_train, y_train, x_test, y_test))

trainingError = list()
testingError = list()
'''
for m in range(1, 200):
    pcaTrain, pcaTest, percentageExplainedPC = pca(x_train, x_test, nComponents=m, plot=False)
    results = linear_regression(pcaTrain, y_train, pcaTest, y_test)
    trainingError.append(results[0])
    testingError.append(results[1])

plotTrainTestPerformance(trainingError, testingError, 'Principal Components')
'''
train_x, test_x = vectors_to_matrices(x_train), vectors_to_matrices(x_test)
print(train_x.shape, test_x.shape)
cnn(train_x, y_train, test_x, y_test)