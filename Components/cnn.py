import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from Components import matrices_to_tensors
from Visualisation import plotTrainTestPerformance

## Architecture
class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cl1 = newLayer(1,4)
        self.cl2 = newLayer(4,4)
        self.linear1 = Sequential(Linear(64, 10))
    # Defining the forward pass
    def forward(self, x):
        x = self.cl1(x)
        x = self.cl2(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        return x


def newLayer(input, output):
    layer = Sequential(
            Conv2d(input, output, kernel_size=2, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2))
    return layer


def cnn(train_x, train_y, test_x, test_y, epochs, learningRate):
    #define the model
    model = Net()
    model = model.float()
    #define the optimizer
    optimizer = Adam(model.parameters(), lr=learningRate)
    #defining the loss function
    criterion = CrossEntropyLoss()
    # transform data into tensors
    train_x, train_y, test_x, test_y = matrices_to_tensors(train_x, train_y, test_x, test_y)
    # training and getting accuracy on test set
    loss_list = []
    acc_list_train = []
    acc_list_test = []
    num_epochs = epochs
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

        total = train_y.size(0)
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

    plotTrainTestPerformance(acc_list_train, acc_list_test, "num_epochs")
    print("Last accuracy test: " + str(acc_list_test[num_epochs-1]))

def crossvalidationCNN():
    pass