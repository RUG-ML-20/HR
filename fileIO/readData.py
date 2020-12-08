import numpy as np
import matplotlib.pyplot as plt

def load():
    file = "data/mfeat-pix.txt"
    mfeat_pix = np.loadtxt(file, dtype='i', delimiter=',')
    
    #print(mfeat_pix[0])
    fig, ax = plt.subplots(10, 10,sharex='col', sharey='row')    
    for i in range(0,10):
        for j in range(0,10):
            pic = np.array(mfeat_pix[200 * (i)+j][:])
            picmatreverse = np.zeros((15,16))
            #the filling is done column wise
            picmatreverse = -pic.reshape(15,16, order = 'F')            
            picmat = np.zeros((15,16))
            for k in range(0,15):
                picmat[:][k] = picmatreverse[:][15-(k+1)]
            picmat = np.transpose(picmat)
            ax[i, j].pcolor(picmat)
        
    plt.show()

            


    pass