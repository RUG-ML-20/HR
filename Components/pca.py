import numpy as np
import pandas as pd 
import random as rd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt

def pca(train,test, nComponents=2):
    #find transformation settings for the data(Standard scalar will transform data so mean = 0 and var = 1)
    scaler = preprocessing.StandardScaler()
    scaler.fit(train) #fit using only train data

    #apply to train and test
    tTrain = scaler.transform(train)
    tTest = scaler.transform(test)

    pca = PCA(n_components=nComponents)
    pca.fit(tTrain)
    pcaTrain = pca.transform(tTrain)
    pcaTest = pca.transform(tTest)

    perVar = np.round(pca.explained_variance_ratio_*100,decimals=1)
    labels = ['PC' + str(x) for x in range(1,len(perVar)+1)]

    plt.bar(x=range(1,len(perVar) + 1), height=perVar, tick_label=labels)
    plt.ylabel('Percentage of Explained Varience')
    plt.xlabel('Principal Component')
    plt.title('Scree Plot')
    plt.show()

        
    return pcaTrain, pcaTest, perVar

