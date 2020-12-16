import numpy as np
import matplotlib.pyplot as plt
from Components.transformations import vector_to_matrix


def plotNumbers(data):
    fig, ax = plt.subplots(10, 10, sharex='col', sharey='row')
    for i in range(0, 10):
        for j in range(0, 10):
            pic = np.array(data[200 * (i)+j][:])
            picmat = vector_to_matrix(pic)
            ax[i, j].pcolor(picmat, cmap='Greys')
    plt.show()

def plotTrainTestPerformance(train, test):
    plt.plot(train)
    plt.plot(test)
    plt.show()
    pass