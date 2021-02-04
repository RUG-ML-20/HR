import numpy as np
import random as rd
from FileIO import load
from Components import *
from Visualisation import plotNumbers, plotTrainTestPerformance, plotWrongDigits, tsne_plots
import sys
from tqdm import tqdm

# ----------loading data-------------
x_train, y_train, x_test, y_test = load(.5, plot=False)

# show division of data set in training vs testing
# data_analysis(x_train, y_train, x_test, y_test)

# ---------linear regression----------
# print('linear regression scores')
# print(f'training: {crossval_LR(x_train,y_train)}')
# print(f'test: {crossval_LR(x_test,y_test)}')
# trainingError = list()
# testingError = list()
# print('PCA')
# print('finding optimal number of principal components')
# best = 0
# best_m = 0
# for m in tqdm(range(1, 200 + 1)):
#     pcaTrain, pcaTest, _ = pca(x_train, x_test, nComponents=m)
#     results_train = crossval_LR(pcaTrain, y_train)
#     results_test = crossval_LR(pcaTest, y_test) 
#     if results_train > best:
#         best = results_train
#         best_m = m
#     trainingError.append(results_train)
#     testingError.append(results_test)
# print(f'optimal number of principal components: {best_m}')
# print(f'score: {best}')
# plotTrainTestPerformance(trainingError, testingError, 'Principal Components')




#---------Set random parameters ------------#
# baseline model with random parameters based on lit (blog) research
#print(randomParameters())

#------- print tSNE plots----------

#test_model(x_train, y_train, x_test, y_test)
# model, _ = train_cnn(x_train, y_train)
# acc,_,_,_ = eval_cnn(model, x_test, y_test, tSNE_list=False)
# print(acc)
# print_layers(model, x_test[5], y_test[5])
# acc,_,_,_, feature_vectors = eval_cnn(model, x_test, y_test, tSNE_list=True)
# print(acc)
# tsne_plots(x_test, y_test, feature_vectors)

# ------------cross-validation----------------

train, test, m, change, saveLocation = crossvalidationCNN(x_train, y_train, 10)
plotTrainTestPerformance(train, test, change, saveLocation, x_values=m)

# ---------Testing model----------

# test_model(x_train,y_train, x_test, y_test)

