import numpy as np
from FileIO import load
from Components import *
from Visualisation import plotNumbers, plotTrainTestPerformance

# loading data
x_train, y_train, x_test, y_test = load(.5, plot=False)

# cross-validation
train, test, m, change = crossvalidationCNN(x_train, y_train, 10)
plotTrainTestPerformance(train, test, change, x_values=m)

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
'''
x_train, x_test = vectors_to_matrices(x_train), vectors_to_matrices(x_test)
# print(x_train.shape, x_test.shape)
accuracy = []
for i in range(5):
    model, loss = train_cnn(x_train, y_train)
    acc_test = eval_cnn(model, x_test, y_test)
    accuracy.append(acc_test)
    print('model', i+1, 'accuracy =', acc_test)
print('average testing accuracy over', len(accuracy), 'trained models =', sum(accuracy)/len(accuracy))
'''
