import numpy
from fileIO.readData import *

train_x, train_y, test_x, test_y = load()
linear_regression(train_x, train_y, test_x, test_y)

#push test niclas
#push test jan