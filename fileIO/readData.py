import numpy as np

#shuffles the dataset randomly, so the order of digits is different every time
def shuffling(X_train, y_train):
    shuffled_rows = np.random.permutation(X_train.shape[0])
    return X_train[shuffled_rows], y_train[shuffled_rows]

def createLabels(data):
    labels = list()
    class_label = 0
    for i in range(0, len(data)):
        labels.append(class_label)
        if i % 200 == 0 and i != 0:
            class_label += 1
    return labels

#load a split data
def load(split = .5):
    file = "data/mfeat-pix.txt"
    data = np.loadtxt(file, dtype='i', delimiter=',')
    # store labels for corresponding digits and shuffle
    labels = np.array(createLabels(data))
    data_X, data_Y = shuffling(data, labels)
    # split into training testing, atm 80% for training 20% for testing
    train_test_split = int(split * data.shape[0])

    train_X = np.array(data_X[:train_test_split])
    test_X = np.array(data_X[train_test_split:])
    train_Y = np.array(data_Y[:train_test_split])
    test_Y = np.array(data_Y[train_test_split:])

    return train_X, train_Y, test_X, test_Y

