import numpy as np
from FileIO import load
from Components import *
from Visualisation import plotNumbers, plotTrainTestPerformance


x_train, y_train, x_test, y_test = load(.5)

# x = [[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4],[5,5,5,5],[6,6,6,6],[7,7,7,7],[8,8,8,8],[9,9,9,9],[10,10,10,10],[11,11,11,11],[12,12,12,12],[13,13,13,13],[14,14,14,14],[15,15,15,15],[16,16,16,16]]
# y = [111,222,333,444,555,666,777,888,999,1000,1111,1222,1333,1444,1555,1666]
# x = np.array(x)
# y = np.array(y)
# crossvalidationCNN(x, y, 4)
#plotNumbers(x_train)
train, test = crossvalidationCNN(x_train, y_train, 10)
train = get_averages(train)
test = get_averages(test)
plotTrainTestPerformance(train, test, 'epochs')


#data_analysis(x_train, y_train, x_test, y_test)

# print(linear_regression(x_train,y_train, x_test, y_test))


# trainingError = list()
# testingError = list()

# for m in range(1, 200):
#     pcaTrain, pcaTest, percentageExplainedPC = pca(x_train, x_test, nComponents=m, plot=False)
#     results = linear_regression(pcaTrain, y_train, pcaTest, y_test)
#     trainingError.append(results[0])
#     testingError.append(results[1])

# plotTrainTestPerformance(trainingError, testingError, 'Principal Components')

# x_train, x_test = vectors_to_matrices(x_train), vectors_to_matrices(x_test)
# print(x_train.shape, x_test.shape)
# cnn(x_train, y_train, x_test, y_test, 100, 0.07)
