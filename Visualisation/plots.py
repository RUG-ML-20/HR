import numpy as np
import matplotlib.pyplot as plt
from Components.transformations import vector_to_matrix


def plotNumbers(data):
    fig, ax = plt.subplots(10, 10, sharex='col', sharey='row')
    for i in range(0, 10):
        for j in range(0, 10):
            pic = np.array(data[200 * (i) + j][:])
            picmat = vector_to_matrix(pic)
            ax[i, j].pcolor(picmat, cmap='Greys')
            ax[i, j].axes.xaxis.set_visible(False)
            ax[i, j].axes.yaxis.set_visible(False)
    plt.show()


def plotTrainTestPerformance(train, test, change, x_values=[]):
    if not x_values:
        plt.plot(train)
        plt.plot(test)
    else:
        plt.plot(x_values, train)
        plt.plot(x_values, test)
    plt.title('Training vs Testing error')
    plt.xlabel(change)
    plt.ylabel('Accuracy')
    plt.legend(['Training', 'Testing'], loc=4)
    plt.show()
    pass
