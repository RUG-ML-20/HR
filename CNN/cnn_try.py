import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
import torch.nn.functional as F


## Architecture
class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 4, kernel_size=2, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(4, 4, kernel_size=2, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2)
        )
        self.linear_layers = Sequential(
            Linear(64, 10)
        )
    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

def cnn(x_train, y_train, x_test, y_test):
    #define the model
    model = Net()
    model = model.float()
    #define the optimizer
    optimizer = Adam(model.parameters(), lr=0.07)
    #defining the loss function
    criterion = CrossEntropyLoss()
    # converting the data into torch format
    train_x = x_train.reshape(1600, 1, 16, 15)
    train_x = torch.from_numpy(train_x)
    train_y = y_train.astype(int)
    train_y = torch.from_numpy(train_y)
    train_x = train_x.float()
    x_test = x_test.reshape(400, 1, 16, 15)
    test_x = torch.from_numpy(x_test)
    test_x = test_x.float()
    test_y = y_test.astype(int)
    test_y = torch.from_numpy(test_y)

    # training and getting accuracy on test set
    loss_list = []
    acc_list = []
    num_epochs = 100
    for epoch in range(num_epochs):
        # Here we feed in training data and perform backprop according to the loss
            # Run the forward pass
            outputs = model.forward(train_x)
            loss = criterion(outputs, train_y)
            loss_list.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Track the accuracy on the testing set / testing set is not used for training no backprop
        # based on these outputs
            outputs_test = model.forward(test_x)
            total = test_y.size(0)
            _, predicted = torch.max(outputs_test.data, 1)
            correct = (predicted == test_y).sum().item()
            acc_list.append(correct / total)
            print("Accuracy on test set: " + str((correct/total) * 100))
