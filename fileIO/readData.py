import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def linear_regression(train_x, train_y, test_x, test_y):
    reg = LinearRegression().fit(train_x, train_y)
    accuracy = reg.score(test_x, test_y)
    print("Linear regression accuracy: " + str(accuracy))

#shuffles the dataset randomly, so the order of digits is different every time
def shuffling(X_train, y_train):
    shuffled_rows = np.random.permutation(X_train.shape[0])
    return X_train[shuffled_rows], y_train[shuffled_rows]

# returns the matrix representation of a vector and if plot = True plots the digit
def vector_to_pic_display(pic, label, plot=False):
    picmatreverse = np.zeros((15, 16))
    # the filling is done column wise
    picmatreverse = -pic.reshape(15, 16, order='F')
    picmat = np.zeros((15, 16))
    for k in range(0, 15):
        picmat[:][k] = picmatreverse[:][15 - (k + 1)]
    picmat = np.transpose(picmat)
    picmat = np.flip(picmat, 0)
    picmat = np.flip(picmat, 1)
    if plot:
        fig = plt.pcolor(picmat, cmap='Greys')
        plt.show()
        print("Label = " + str(label))
    return picmat

def plotNumbers(data):
    fig, ax = plt.subplots(10, 10, sharex='col', sharey='row')
    for i in range(0, 10):
        for j in range(0, 10):
            pic = np.array(data[200 * (i)+j][:])
            picmatreverse = np.zeros((15,16))
            #the filling is done column wise
            picmatreverse = -pic.reshape(15,16, order = 'F')            
            picmat = np.zeros((15, 16))
            for k in range(0, 15):
                picmat[:][k] = picmatreverse[:][15-(k+1)]
            picmat = np.transpose(picmat)
            picmat = np.flip(picmat, 0)
            picmat = np.flip(picmat, 1)
            ax[i, j].pcolor(picmat, cmap='Greys')
            
    plt.show()

'''
load the data, Arguments are plot -> set to True if you want a plot of some example digits
and vector_representation -> when false this function returns the data in matrix format, otherwise 
data is returned in vectors
'''
def load(plot=False, vector_representation = False):
    file = "data/mfeat-pix.txt"
    mfeat_pix = np.loadtxt(file, dtype='i', delimiter=',')
    if plot:
        plotNumbers(mfeat_pix)
    # store labels for corresponding digits and shuffle
    data = mfeat_pix
    labels = []
    class_label = 0
    for i in range(0, len(mfeat_pix)):
        labels.append(class_label)
        if i % 200 == 0 and i != 0:
            class_label += 1
    labels_np = np.array(labels)
    data_X, data_Y = shuffling(data, labels_np)
    # split into training testing, atm 80% for training 20% for testing
    train_test_split = int(.8 * data.shape[0])
    # data is returned in matrix format
    if not vector_representation:
        # convert the vectors in matrices and store
        pic_data_x = {}
        for i in range(0, len(data_X)):
            pic_data_x[i] = vector_to_pic_display(data_X[i], data_Y[i], False)

        # store training and testing matrices with indexes
        train_X = []
        test_X = []
        for i in range(0, train_test_split):
            train_X.append(pic_data_x[i])
        for i in range(train_test_split, len(data_X)):
            test_X.append(pic_data_x[i])
        # convert to np arrays and store the corresponding labels
        train_X = np.array(train_X)
        train_Y = np.array(data_Y[:train_test_split])
        test_X = np.array(test_X)
        test_Y = np.array(data_Y[train_test_split:])
    # data is returned in vector format
    else:
        train_X = np.array(data_X[:train_test_split])
        test_X = np.array(data_X[train_test_split:])
        train_Y = np.array(data_Y[:train_test_split])
        test_Y = np.array(data_Y[train_test_split:])

    return train_X, train_Y, test_X, test_Y
