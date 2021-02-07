import numpy as np
import matplotlib.pyplot as plt
from Components.transformations import vector_to_matrix
from sklearn.manifold import TSNE
import seaborn as sns


def plot_hidden_layers(x, n_channels, layer):
    data = x[0].detach().numpy()
    fig, ax = plt.subplots(int(n_channels/2)+1, int(n_channels/2), sharex='col', sharey='row')
    fig.suptitle(layer)
    for j in range(0, int(n_channels/2)+1):
        for i in range(0, int(n_channels/2)):
            pic = data[i]
            pic = np.flip(pic, 0)
            ax[j, i].pcolor(pic, cmap='gist_gray')
            ax[j, i].axes.xaxis.set_visible(False)
            ax[j, i].axes.yaxis.set_visible(False)
    plt.show()

def plotNumbers(data):
    fig, ax = plt.subplots(10, 10, sharex='col', sharey='row')
    for i in range(0, 10):
        for j in range(0, 10):
            pic = np.array(data[200 * (i) + j][:])
            picmat = vector_to_matrix(pic)
            ax[i, j].pcolor(picmat, cmap='gist_gray')
            ax[i, j].axes.xaxis.set_visible(False)
            ax[i, j].axes.yaxis.set_visible(False)
    plt.show()
    fig.savefig('digits.png')


def plotWrongDigits(x, predicted, y, filename, num):
    numDigits = len(x)
    fig, ax = plt.subplots(5, 10, sharex='col', sharey='row')
    for i in range(50):
        col = i % 10
        row = int(i/10)
        if i < numDigits:
            pic = np.array(x[i][:])
            ax[row, col].pcolor(pic, cmap='gist_gray')
            ax[row, col].axes.xaxis.set_ticks([])
            ax[row, col].set_xlabel('pred: '+str(predicted[i])+'\nlabel: '+str(y[i]))
            ax[row, col].axes.yaxis.set_visible(False)
        else:
            ax[row, col].axis('off')
    plt.tight_layout()
    plt.savefig(f"{filename}/plot_{num}.jpg", bbox_inches='tight', dpi=150)


def plotTrainTestPerformance(train, test, change,filename = None, x_values=[]):
    train[:] = [1 - x for x in train]
    test[:] = [1 - x  for x in test]
    if not x_values:
        plt.plot(train)
        plt.plot(test)
    else:
        plt.plot(x_values, train)
        plt.plot(x_values, test)
    axes = plt.gca()
    axes.set_ylim([0,1])
    plt.xlabel(change)
    plt.ylabel('Error rate')
    plt.legend(['Training', 'Testing'], loc=1)
    if filename:
        plt.savefig(f'{filename}/crossValidationPlot.png')
    else:
        plt.show()


def tsne_plots(x_test, y_test, feature_vectors):
    tSNE = TSNE(n_components=2, perplexity=40, random_state=5, n_iter=2000)
    embed_digits = tSNE.fit_transform(x_test)
    embed_digits_cnn = tSNE.fit_transform(feature_vectors.detach().numpy())

    fig, axes = plt.subplots(1, 2)
    fig.suptitle('Embedding of digits without preprocessing and after the convolutional layers')
    sns.scatterplot(x=embed_digits[:, 0], y=embed_digits[:, 1], hue=y_test+1, palette='tab10', ax=axes[0])
    axes[0].set_title("Digits without processing")
    sns.scatterplot(x=embed_digits_cnn[:, 0], y=embed_digits_cnn[:, 1], hue=y_test+1, palette='tab10', ax=axes[1])
    axes[1].set_title("Digits after convolutions")
    plt.show()