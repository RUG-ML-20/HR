import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.optim import Adam, SGD
from Components import matrices_to_tensors
from Visualisation import plotTrainTestPerformance
import numpy as np

## Architecture
class Net(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, layers):
        super(Net, self).__init__()
        self.input_dim = input_dim
        self.out_dim = output_dim
        self.hidden_dim = hidden_dim
        current_dim = input_dim
        self.layers = nn.ModuleList()
        self.to_linear = None
        for hdim in range(layers-1):
            self.layers.append(newLayer(current_dim, hidden_dim))
            current_dim = hidden_dim
        x = torch.randn(15,16).view(-1,1,15,16)
        self.convs(x)
        self.linear1 = nn.Linear(self.to_linear, 10)


    def convs(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
        if self.to_linear is None:
            self.to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.to_linear)
        x = self.linear1(x)
        return x

def newLayer(input, output):
    layer = nn.Sequential(
            nn.Conv2d(input, output, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )

    return layer


def cnn(train_x, train_y, test_x, test_y, epochs, learningRate, dimensions):
    data = matrices_to_tensors(train_x, train_y, test_x, test_y)
    model_input = dimensions[0]
    model_output = dimensions[1]
    hidden_output = dimensions[2]
    hidden_layers = dimensions[3]

    model = Net(model_input,model_output,hidden_output,hidden_layers)
    model = model.float()
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
    model_dimensions = (1,10,4,3) #(input channels, output channels, hiddenlayer outputs, number of hidden layers)
    for fold in range(0, k):
        print('fold: ',fold+1)
        train_x, train_y, test_x, test_y = get_fold(folds_x, folds_y, fold)
        results = cnn(train_x, train_y, test_x, test_y, 100, 0.07, model_dimensions)
        accTrain.append(results[0])
        accTest.append(results[1])
        loss.append(results[2])
        print('final accuracy reading')
        print('train: ',accTrain[fold][-1],'test: ',accTest[fold][-1])
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



    