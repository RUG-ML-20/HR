import numpy as np


# shuffles the dataset randomly, so the order of digits is different every time
def shuffling(X_train, y_train):
    np.random.seed(30)
    shuffled_rows = np.random.permutation(X_train.shape[0])
    return X_train[shuffled_rows], y_train[shuffled_rows]


def createLabels(data):
    labels = list()
    class_label = 0
    for i in range(1, len(data)+1):
        labels.append(class_label)
        if i % 200 == 0 and i != 0:
            class_label += 1
    return labels

def distribute_digits_split(data_X, data_Y, split):
    split = int(200 * split)
    train_x = []
    train_y =[]
    test_x = []
    test_y = []
    for i in range(0,10):
        train_x.append(data_X[i*200: i*200+split])
        train_y.append(data_Y[i*200: i*200+split])
        test_x.append(data_X[i*200+split: i*200+200])
        test_y.append(data_Y[i*200+split: i*200+200])

    return np.array(train_x).reshape(-1,240), np.array(train_y).flatten(), \
           np.array(test_x).reshape(-1,240), np.array(test_y).flatten()

# load a split data
def load(split=.5):
    file = "data/mfeat-pix.txt"
    data = np.loadtxt(file, dtype='i', delimiter=',')
    # store labels for corresponding digits and shuffle
    labels = np.array(createLabels(data))
    train_x, train_y, test_x, test_y = distribute_digits_split(data, labels, split)

    train_x, train_y = shuffling(train_x, train_y)
    test_x, test_y = shuffling(test_x, test_y)

    return train_x, train_y, test_x, test_y
