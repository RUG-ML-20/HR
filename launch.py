import numpy
from fileIO.readData import *
from Components.pca import pca

train_x, train_y, test_x, test_y = load(load(plot=False, vector_representation=False)
pcaTrain, pcaTest, percentageExplainedPC  = pca(train_x, test_x, nComponents=60)

linear_regression(train_x, train_y, test_x, test_y)

linear_regression(pcaTrain, train_y, pcaTest, test_y)

#push test niclas
#push test jan
