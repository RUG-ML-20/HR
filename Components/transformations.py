import numpy as np
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