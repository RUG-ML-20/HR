import numpy as np
import pandas as pd
import random as rd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import matplotlib.pyplot as plt
from tqdm import tqdm


def pca(train, test, nComponents=2, plot=False):
    # find transformation settings for the data
    # (standard scalar will transform data so mean = 0 and var = 1)
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
    # amount of explained variance per principal component
    perVar = np.round(pca.explained_variance_ratio_ * 100, decimals=1)

    if plot:
        labels = ['PC' + str(x) for x in range(1, len(perVar) + 1)]
        plt.bar(x=range(1, len(perVar) + 1), height=perVar, tick_label=labels)
        plt.ylabel('Percentage of Explained Variance')
        plt.xlabel('Principal Component')
        plt.show()
    return pcaTrain, pcaTest, perVar


def linear_regression(train_x, train_y, test_x, test_y):
    reg = LinearRegression().fit(train_x, train_y)
    trainAccuracy = reg.score(train_x, train_y)
    testAccuracy = reg.score(test_x, test_y)
    return trainAccuracy, testAccuracy


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
    for y in ys:
        occurrences[int(y)] += 1
    return occurrences


def randomParameters():
    epochs = [50, 60, 70, 80, 90, 100]
    epoch = rd.choice(epochs)
    learningRates = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    learningRate = rd.choice(learningRates)
    l2s = [0 + (x/10) for x in range(40)]
    l2 = rd.choice(l2s)
    batchSizes = [8, 16, 32, 64, 128, 256]
    batchSize = rd.choice(batchSizes)
    return epoch, learningRate, l2, batchSize


def crossval_LR(x, y, k, pca_run=False):
    total = x.shape[0]
    k = split_check(total, k)
    bin_size = int(total / k)
    folds_x = np.array(np.array_split(x, k))
    folds_y = np.array(np.split(y, k))
    acc_train_m = list()
    acc_test_m = list()
    m_list = list()
    change = "principal components"
    best_m = 0
    best_m_acc = 0
    if pca_run:
        m_range = np.arange(1, 241, 1)
    else:
        m_range = range(0, 1)
    print(f'training and evaluating {k * len(m_range)} models')
    for m in tqdm(m_range, desc='PCs', position=0):  # loop over given m settings
        acc_train = list()
        acc_test = list()
        for fold in range(0, k):  # train a new model for each fold and for each m
            train_x, train_y, test_x, test_y = get_fold(folds_x, folds_y, fold)
            if pca_run:
                train_x, test_x, _ = pca(train_x, test_x, nComponents=m)
            reg = LinearRegression().fit(train_x, train_y)
            results_train = reg.score(train_x, train_y)
            results_test = reg.score(test_x, test_y)
            acc_train.append(results_train)
            acc_test.append(results_test)
        mean_train_acc = round(np.mean(acc_train), 4)
        mean_test_acc = round(np.mean(acc_test), 4)
        acc_train_m.append(mean_train_acc)
        acc_test_m.append(mean_test_acc)
        if mean_test_acc > best_m_acc:
            best_m_acc = mean_test_acc
            best_m = m
    return acc_train_m, acc_test_m, best_m, best_m_acc


# returns the folds in separate training and testing data
# the choice of which fold is used for testing data is indicated by the index n
def get_fold(folds_x, folds_y, n):
    test_x = folds_x[n]
    test_y = folds_y[n]
    temp = np.repeat(True, folds_x.shape[0])
    temp[n] = False
    train_x = folds_x[temp]
    train_y = folds_y[temp]
    train_x = np.concatenate(train_x, axis=0)
    train_y = np.concatenate(train_y, axis=0)
    return train_x, train_y, test_x, test_y


# helper function to make sure data can be split up into k folds without causing shape issues,
# if there is an issue, the closest number to k that will not cause problems will be chosen
# with a preference to the higher number.
def split_check(n, k):
    if n % k == 0:
        return k
    u = 1
    while n % (k + u) != 0 and (k - u < 2 or n % (k - u) != 0):
        u += 1
    if n % (k + u) == 0:
        nk = k + u
    elif n % (k - u) == 0:
        nk = k - u
    print(f'Warning: current K={k} for K-fold cross-validation would not divide folds correctly')
    print(f'the new k: {nk} was chosen instead')
    return nk


def test_LR(x_train, y_train, x_test, y_test, pca_run=False, m=0):
    if pca_run:
        x_train, x_test, _ = pca(x_train, x_test, nComponents=m)
    accuracy = []
    for i in range(10):
        model = LinearRegression().fit(x_train, y_train)
        acc_test = model.score(x_test, y_test)
        accuracy.append(acc_test)
    return round(np.mean(accuracy), 4)
