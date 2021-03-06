import numpy as np
import sys
import re


# shuffles the dataset randomly, so the order of digits is different every time
def shuffling(X_train, y_train):
    np.random.seed(30)
    shuffled_rows = np.random.permutation(X_train.shape[0])
    return X_train[shuffled_rows], y_train[shuffled_rows]


def createLabels(data):
    labels = list()
    class_label = 0
    for i in range(1, len(data) + 1):
        labels.append(class_label)
        if i % 200 == 0 and i != 0:
            class_label += 1
    return labels


def distribute_digits_split(data_X, data_Y, split):
    split = int(200 * split)
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for i in range(0, 10):
        train_x.append(data_X[i * 200: i * 200 + split])
        train_y.append(data_Y[i * 200: i * 200 + split])
        test_x.append(data_X[i * 200 + split: i * 200 + 200])
        test_y.append(data_Y[i * 200 + split: i * 200 + 200])

    return np.array(train_x).reshape(-1, 240), np.array(train_y).flatten(), \
           np.array(test_x).reshape(-1, 240), np.array(test_y).flatten()


# load and split data
def load(split=.5, plot=False):
    file = "data/mfeat-pix.txt"
    data = np.loadtxt(file, dtype='i', delimiter=',')
    if plot:
        from Visualisation.plots import plotNumbers
        plotNumbers(data)

    # store labels for corresponding digits and shuffle
    labels = np.array(createLabels(data))
    train_x, train_y, test_x, test_y = distribute_digits_split(data, labels, split)
    train_x, train_y = shuffling(train_x, train_y)
    test_x, test_y = shuffling(test_x, test_y)
    return train_x, train_y, test_x, test_y


def save_model(location, model, accuracy):
    sys.stdout = open(f'{location}/summary.txt', "w")
    print(f"model accuracy: {accuracy}")
    print(model)
    sys.stdout.close()


def get_run_number(filename):
    with open(filename, 'r+') as f:
        num = f.readline()
        num = re.sub(num, f"{int(num) + 1}", num)
        f.seek(0)
        f.write(num)
        f.truncate()
    return int(num)


def save_accuracies(filename, arr):
    array = arr
    with open(f"{filename}/accuracies.txt", "w") as output:
        output.write(str(array))


def save_best_m(filename, changed, m, acc):
    with open(f"{filename}/optimal_m.txt", "w") as output:
        output.write(f'optimal {changed} value: {m} with an accuracy of {acc}')


def save_accuracies_sum(filename, m, train, test):
    with open(f"{filename}/overall_accuracies.txt", "w") as output:
        output.write(f'm,train,test\n')
        for i in range(0, len(m)):
            output.write(f'{round(m[i], 4)},{train[i]},{test[i]}\n')


def read_results(filename):
    file1 = open(f"{filename}/overall_accuracies.txt", "r")
    lines = file1.readlines()
    m = list()
    train = list()
    test = list()
    # traverse through lines one by one
    for line in lines:
        if not line.split(',')[0] == "m":
            split = line.split(',')
            m.append(float(split[0]))
            train.append(float(split[1]))
            test.append(float(split[2]))
    file1.close()
    return m, train, test
