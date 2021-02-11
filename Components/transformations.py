import numpy as np
import torch


# returns the matrix representation of a vector and if plot = True plots the digit
def vector_to_matrix(pic):
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


def matrices_to_tensors(x, y):
    x = x.reshape(x.shape[0], 1, 16, 15)
    x = torch.from_numpy(x)
    y = y.astype(int)
    y = torch.from_numpy(y)
    x = x.float()
    return x, y


def get_averages(arr):
    new_list = list()
    for res in arr:
        new_list.append(np.mean(res))
    return new_list
