import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.optim import Adam, SGD
from Components import matrices_to_tensors, labels_to_vectors
from Visualisation import plotTrainTestPerformance
import numpy as np

## Architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.to_linear = None
        # sets the sequential layers
        self.sequential_layers_conv = sequential_layers_conv()
        # set a random image to feed into convs to calculate output dim
        x = torch.randn(15,16).view(-1,1,15,16)
        self.convs(x)
        # here you could print how many units there are before the linear layers (might be helpful)
        #print('to_linear')
        #print(self.to_linear)
        self.linear = sequential_layers_linear(self.to_linear)

    def convs(self, x):
        x = self.sequential_layers_conv(x)
        # calculate how many output units there will be after the convolutional layers
        if self.to_linear is None:
            self.to_linear = x.shape[1] * x.shape[2] * x.shape[3]

        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.to_linear)
        x = self.linear(x)
        return x


# specifies the linear layers with the input being calculated before (to_linear)
def sequential_layers_linear(to_linear):
    layer = nn.Sequential(nn.Linear(to_linear, 10))

    return layer


# specify the network architecture here
def sequential_layers_conv():
    layer = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(3,4), stride=1, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, 10, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )

    return layer


def train_cnn(x, y, epochs=100, learningRate = 0.01, l2_weight_decay = 0.01, batch_size = None):
    x, y = matrices_to_tensors(x, y)
    model = Net()
    model = model.float()
    optimizer = Adam(model.parameters(), lr=learningRate, weight_decay=l2_weight_decay)
    criterion = nn.CrossEntropyLoss()
    loss_list = []
    #batch sizes are claculated and sorted, by defaut batchsize is entire dataset
    if not batch_size or batch_size > x.shape[0]:
        batch_size = x.shape[0]
    batch_num = x.shape[0]/batch_size
    x = x.reshape(-1, batch_size, 1, 15, 16)
    y = y.reshape(-1, batch_size)

    for epoch in range(0, epochs):
        # loop over the number of batches feeds in batch_size many images and performs backprob
        # then again and so on
        for i in range(0, int(batch_num)):
            # Here we feed in training data and perform backprop according to the loss
            # Run the forward pass
            outputs = model.forward(x[i])
            y = y.long()
            loss = criterion(outputs, y[i])
            loss_list.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model, loss_list

    

def eval_cnn(model, x, y):
    x,y = matrices_to_tensors(x,y)
    output = model.forward(x)
    total = y.size(0)
    _, predicted = torch.max(output.data, 1)
    correct = (predicted == y).sum().item()
    return correct / total

# def train_cnn(train_x, train_y, epochs=100, learningRate = 0.01, l2_weight_decay = 0.01, batch_size = 1):
#     train_x, train_y, test_x, test_y = matrices_to_tensors(train_x, train_y, test_x, test_y)
#     model = Net()
#     model = model.float()

#     optimizer = Adam(model.parameters(), lr=learningRate, weight_decay=l2_weight_decay)
#     criterion = nn.CrossEntropyLoss()
#     acc_list_train = []
#     acc_list_test = []
#     loss_list = []
#     # here you can choose the batch size (currently its set to how all images at once)
#     batch_num = train_x.shape[0]/batch_size
#     train_x = train_x.reshape(-1, batch_size, 1, 15, 16)
#     train_y = train_y.reshape(-1, batch_size)
#     print(train_y.shape)

#     for epoch in range(epochs):
#         correct = 0
#         # loop over the number of batches feeds in batch_size many images and performs backprob
#         # then again and so on
#         for i in range(0, int(batch_num)):
#             # Here we feed in training data and perform backprop according to the loss
#             # Run the forward pass
#             outputs = model.forward(train_x[i])
#             train_y = train_y.long()
#             loss = criterion(outputs, train_y[i])
#             loss_list.append(loss.item())

#             # Backprop and perform Adam optimisation
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             predicted = torch.max(outputs.data, 1)
#             correct += (predicted[1] == train_y[i]).sum()

#         total = int(batch_num * batch_size)
#         acc_list_train.append(float(correct) / total)

#         # Track the accuracy on the testing set / testing set is not used for training no backprop
#         # based on these outputs
#         outputs_test = model.forward(test_x)
#         total = test_y.size(0)
#         _, predicted = torch.max(outputs_test.data, 1)
#         correct = (predicted == test_y).sum().item()
#         acc_list_test.append(correct / total)
    
#     return model





# def run_model(model, data, epochs, learningRate, l2_weight_decay, batch_size):
#     optimizer = Adam(model.parameters(), lr=learningRate, weight_decay=l2_weight_decay)
#     criterion = nn.CrossEntropyLoss()
#     acc_list_train = []
#     acc_list_test = []
#     loss_list = []
#     # here you can choose the batch size (currently its set to how all images at once)
#     batch_num = train_x.shape[0]/batch_size
#     train_x = train_x.reshape(-1, batch_size, 1, 15, 16)
#     train_y = train_y.reshape(-1, batch_size)
#     print(train_y.shape)

#     for epoch in range(epochs):
#         correct = 0
#         # loop over the number of batches feeds in batch_size many images and performs backprob
#         # then again and so on
#         for i in range(0, int(batch_num)):
#             # Here we feed in training data and perform backprop according to the loss
#             # Run the forward pass
#             outputs = model.forward(train_x[i])
#             train_y = train_y.long()
#             loss = criterion(outputs, train_y[i])
#             loss_list.append(loss.item())

#             # Backprop and perform Adam optimisation
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             predicted = torch.max(outputs.data, 1)
#             correct += (predicted[1] == train_y[i]).sum()

#         total = int(batch_num * batch_size)
#         acc_list_train.append(float(correct) / total)

#         # Track the accuracy on the testing set / testing set is not used for training no backprop
#         # based on these outputs
#         outputs_test = model.forward(test_x)
#         total = test_y.size(0)
#         _, predicted = torch.max(outputs_test.data, 1)
#         correct = (predicted == test_y).sum().item()
#         acc_list_test.append(correct / total)

#     return acc_list_train, acc_list_test, loss_list

def crossvalidationCNN(x, y, k):
    #setup the kfold split
    total = x.shape[0]
    k = split_check(total, k)
    bin_size = int(total / k)
    folds_x = np.array(np.array_split(x, k))
    folds_y = np.array(np.split(y, k))
    acc_train_m = list()
    acc_test_m = list()
    loss = list()

    #define m range in this case m corresponds with epochs
    #to change what is going to vary with m, mention in the train_cnn function 
    #eg. learning_rate = m, batch_size = m ...
    start = 1
    stop = 100
    step = 1

    wd = 0.021 #value extrapolated from 10 * 0.02 (value per iteration) + 0.001 (initial value)

    for m in range(start, stop + 1, step):# loop over given m settings
        print(m)
        acc_train = list()
        acc_test = list()
        for fold in range(0, k): # train a new model for each fold and for each m
            train_x, train_y, test_x, test_y = get_fold(folds_x, folds_y, fold)
            model, loss = train_cnn(train_x, train_y, epochs=m)
            acc_train.append(eval_cnn(model, train_x, train_y))
            acc_test.append(eval_cnn(model, test_x, test_y))
        acc_train_m.append(acc_train)
        acc_test_m.append(acc_test)
    return acc_train_m, acc_test_m

'''
combines the split up folds into training and testing data. The choice of which fold
is used for testing data is indicated by the index n
'''
def get_fold(folds_x, folds_y, n): 
    test_x = folds_x[n]
    test_y = folds_y[n]
    temp = np.repeat(True, folds_x.shape[0])
    temp[n] = False
    train_x = folds_x[temp]
    train_y = folds_y[temp]
    train_x = np.concatenate(train_x, axis = 0)
    train_y = np.concatenate(train_y, axis = 0)
    return train_x, train_y, test_x, test_y

'''
helper function to make sure data can be split up into k folds without caising shape issues,
if there is an issue, the closest number to k that will not cause problems will be chosen
with a preference to the higher number.
'''
def split_check(n, k):
    if n % k == 0:
        return k
    
    u = 1
    while n % (k + u) != 0 and (k - u < 2 or n % (k - u) != 0):
        u += 1

    if n % (k + u) == 0:
        nk = k + u
    elif n % (k - u) == 0:
        nk = k - u

    print(f'Warning: current K={k} for K-fold cross-validation would not divide folds correctly')
    print(f'the new k: {nk} was chosen instead')
    return nk


    