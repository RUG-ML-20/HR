import numpy as np
import pandas as pd
import random as rd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
import matplotlib.pyplot as plt



def pca(train, test, nComponents=2, plot=False):
    # find transformation settings for the data(Standard scalar will transform data so mean = 0 and var = 1)
    scaler = preprocessing.StandardScaler()
    scaler.fit(train)  # fit using only train data

    # apply to train and test
    tTrain = scaler.transform(train)
    tTest = scaler.transform(test)

    # create instance of pca model and train on train data
    pca = PCA(n_components=nComponents)
    pca.fit(tTrain)
    pcaTrain = pca.transform(tTrain)
    pcaTest = pca.transform(tTest)  # apply to test data
    perVar = np.round(pca.explained_variance_ratio_ * 100,
                      decimals=1)  # amount of explained variance per principal component

    if plot:
        labels = ['PC' + str(x) for x in range(1, len(perVar) + 1)]
        plt.bar(x=range(1, len(perVar) + 1), height=perVar, tick_label=labels)
        plt.ylabel('Percentage of Explained Varience')
        plt.xlabel('Principal Component')
        plt.title('Scree Plot')
        plt.show()
    return pcaTrain, pcaTest, perVar


def linear_regression(train_x, train_y, test_x, test_y):
    reg = LinearRegression().fit(train_x, train_y)
    trainAccuracy = reg.score(train_x, train_y)
    testAccuracy = reg.score(test_x, test_y)
    return trainAccuracy, testAccuracy

def crossval_LR(x,y):
    lr = LinearRegression()
    return np.mean(cross_val_score(lr, x, y, cv = 10))
    
def data_analysis(train_x, train_y, test_x, test_y):
    all_xs = np.concatenate((train_x, test_x), axis=0)
    all_ys = np.concatenate((train_y, test_y), axis=0)
    table = list()
    table.append(count_occurrences(all_ys))
    table.append(count_occurrences(train_y))
    table.append(count_occurrences(test_y))
    table = np.array(table)
    titles = ['overall', 'train', 'test']
    frame = pd.DataFrame(table, index=titles, columns=range(0, 10))
    print(frame)
    pass


def count_occurrences(ys):
    occurrences = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    total = ys.size
    for y in ys:
        occurrences[int(y)] += 1

    return occurrences


def randomParameters():
    epochs = [50, 60, 70, 80, 90, 100]
    epoch = rd.choice(epochs)

    learningRates = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    learningRate = rd.choice(learningRates)

    l2s = [0.01 + x for x in range(40)] # not sure about this parameter 
    l2 = rd.choice(l2s)

    batchSizes = [8, 16, 32, 64, 128, 256]
    batchSize = rd.choice(batchSizes)

    return epoch, learningRate, l2, batchSize
