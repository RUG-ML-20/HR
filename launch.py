import numpy as np
from FileIO import load
from Components import pca, linear_regression
from Visualisation import plotNumbers, plotTrainTestPerformance

x_train, y_train, x_test, y_test = load(.8)
print(linear_regression(x_train, y_train, x_test, y_test))

trainingError = list()
testingError = list()
for m in range(1, 200):
    pcaTrain, pcaTest, percentageExplainedPC = pca(x_train, x_test, nComponents=m, plot=False)
    results = linear_regression(pcaTrain, y_train, pcaTest, y_test)
    trainingError.append(results[0])
    testingError.append(results[1])

plotTrainTestPerformance(trainingError, testingError, 'Principal Components')
