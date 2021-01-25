import numpy as np
import matplotlib.pyplot as plt
from Components.transformations import vector_to_matrix
from sklearn.manifold import TSNE
import seaborn as sns

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


def plotTrainTestPerformance(train, test, change,filename, x_values=[]):
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
    plt.savefig(f'{filename}/crossValidationPlot.png')


def tsne_plots(x_test, y_test, feature_vectors):
    tSNE = TSNE(n_components=2, perplexity=30, random_state=30)
    embed_digits = tSNE.fit_transform(x_test)
    embed_digits_cnn = tSNE.fit_transform(feature_vectors.detach().numpy())
    print(embed_digits.shape, embed_digits_cnn.shape)
    print(embed_digits[0], embed_digits[1])
    fig, axes = plt.subplots(1,2)
    fig.suptitle('Embedding of digits without preprocessing and after the convolutional layers')
    sns.scatterplot(x=embed_digits[:, 0], y=embed_digits[:, 1], hue=(y_test+1), palette='tab10', ax=axes[0])
    axes[0].set_title("Digits without processing")
    sns.scatterplot(x=embed_digits_cnn[:, 0], y=embed_digits_cnn[:, 1], hue=(y_test+1), palette='tab10', ax=axes[1])
    axes[1].set_title("Digits after convolutions")
    plt.show()