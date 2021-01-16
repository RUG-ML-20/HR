import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.optim import Adam, SGD
from Components import matrices_to_tensors
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


def cnn(train_x, train_y, test_x, test_y, epochs, learningRate):
    data = matrices_to_tensors(train_x, train_y, test_x, test_y)
    model = Net()
    model = model.float()
    #print(model)
    optimizer = Adam(model.parameters(), lr=learningRate)
    criterion = nn.CrossEntropyLoss()
    results = run_model(model, data, epochs, optimizer, criterion)
    return results

def run_model(model, data, epochs, optimizer, criterion):
    train_x, train_y, test_x, test_y = data
    acc_list_train = []
    acc_list_test = []
    loss_list = []
    
    for epoch in range(epochs):
        # Here we feed in training data and perform backprop according to the loss
        # Run the forward pass
        outputs = model.forward(train_x)
        train_y = train_y.long()
        loss = criterion(outputs, train_y)
        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total = len(train_y)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == train_y).sum().item()
        acc_list_train.append(correct / total)

        # Track the accuracy on the testing set / testing set is not used for training no backprop
        # based on these outputs
        outputs_test = model.forward(test_x)
        total = test_y.size(0)
        _, predicted = torch.max(outputs_test.data, 1)
        correct = (predicted == test_y).sum().item()
        acc_list_test.append(correct / total)

    return acc_list_train, acc_list_test, loss_list

def crossvalidationCNN(train, labels, k):
    total = train.shape[0]
    k = split_check(total, k)
    bin_size = int(total / k)
    folds_x = np.array(np.array_split(train, k))
    folds_y = np.array(np.split(labels, k))
    accTrain = list()
    accTest = list()
    loss = list()
    avg_acc_train = 0
    avg_acc_test = 0
    for fold in range(0, k):
        print('fold: ',fold+1)
        train_x, train_y, test_x, test_y = get_fold(folds_x, folds_y, fold)
        results = cnn(train_x, train_y, test_x, test_y, 100, 0.07)
        accTrain.append(results[0])
        accTest.append(results[1])
        loss.append(results[2])
        print('fold accuracy reading')
        print('train: ',accTrain[fold][-1],'test: ',accTest[fold][-1])
        avg_acc_train += accTrain[fold][-1]
        avg_acc_test += accTest[fold][-1]
    print('Average accuracy reading')
    print('train: ' + str(avg_acc_train/k) + 'test: ' + str(avg_acc_test/k))
    return accTrain, accTest, loss

'''
combines the split up folds into training and testing data. The choice of which fold
is used for testing data is indicated by the index n
ISSUES
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
    while n % k + u != 0:
        if n % k - u != 0:
            nk = k - u
        if n % k + u != 0:
            nk = k + u
            break
        u += 1
    print(f'Warning: current k: {k} for kfold crossvalidation would not divide folds correctly')
    print(f'the new k: {nk} was chosen instead')
    return nk


    