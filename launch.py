import numpy as np
from FileIO import load
from Components import pca, linear_regression, vectors_to_matrices, data_analysis, cnn
from Visualisation import plotNumbers, plotTrainTestPerformance


x_train, y_train, x_test, y_test = load(.8)
data_analysis(x_train, y_train, x_test, y_test)

print(linear_regression(x_train,y_train, x_test, y_test))


trainingError = list()
testingError = list()

for m in range(1, 200):
    pcaTrain, pcaTest, percentageExplainedPC = pca(x_train, x_test, nComponents=m, plot=False)
    results = linear_regression(pcaTrain, y_train, pcaTest, y_test)
    trainingError.append(results[0])
    testingError.append(results[1])

plotTrainTestPerformance(trainingError, testingError, 'Principal Components')

x_train, x_test = vectors_to_matrices(x_train), vectors_to_matrices(x_test)
print(x_train.shape, x_test.shape)
cnn(x_train, y_train, x_test, y_test, 100, 0.07)