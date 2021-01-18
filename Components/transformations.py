import numpy as np
import torch

# returns the matrix representation of a vector and if plot = True plots the digit
def vector_to_matrix(pic):
    picmatreverse = np.zeros((15, 16))
    # the filling is done column wise
    picmatreverse = -pic.reshape(15, 16, order='F')
    picmat = np.zeros((15, 16))
    for k in range(0, 15):
        picmat[:][k] = picmatreverse[:][15 - (k + 1)]
    picmat = np.transpose(picmat)
    picmat = np.flip(picmat, 0)
    picmat = np.flip(picmat, 1)
    return picmat


def vectors_to_matrices(vectors):
    matrices = list()
    for vector in vectors:
        matrices.append(vector_to_matrix(vector))
    return np.array(matrices)

def labels_to_vectors(labels):
    vectors = np.zeros([len(labels), 10], dtype='int')
    for i in range(0, len(labels)):
        vectors[i, labels[i]] = 1

    return torch.from_numpy(vectors)

def matrices_to_tensors(x_train, y_train, x_test, y_test):
    train_x = x_train.reshape(x_train.shape[0], 1, 15, 16)
    train_x = torch.from_numpy(train_x)
    train_y = y_train.astype(int)
    train_y = torch.from_numpy(train_y)
    train_x = train_x.float()
    x_test = x_test.reshape(x_test.shape[0], 1, 15, 16)
    test_x = torch.from_numpy(x_test)
    test_x = test_x.float()
    test_y = y_test.astype(int)
    test_y = torch.from_numpy(test_y)
    return train_x, train_y, test_x, test_y
