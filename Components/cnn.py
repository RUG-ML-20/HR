import torch.nn as nn
from torch.optim import Adam
from Components.functions import *
from Components.transformations import *
from Visualisation import plotWrongDigits, plot_hidden_layers
from FileIO import *
import numpy as np
import os
from tqdm import tqdm


# Architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.to_linear = None

        # Architecture is defined here now
        self.conv_layer_1 = nn.Conv2d(1, 6, kernel_size=(4, 3), stride=1, padding=1)
        self.batch_norm_1 = nn.BatchNorm2d(6)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv_layer_2 = nn.Conv2d(6, 10, kernel_size=4, stride=1, padding=0)
        self.batch_norm_2 = nn.BatchNorm2d(10)
        self.relu2 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # set a random image to feed into convs to calculate output dimension
        x = torch.randn(16, 15).view(-1, 1, 16, 15)
        self.convs(x)
        self.linear = sequential_layers_linear(self.to_linear)

    def convs(self, x):
        x = self.conv_layer_1(x)
        x = self.batch_norm_1(x)
        x = self.relu1(x)
        x = self.conv_layer_2(x)
        x = self.batch_norm_2(x)
        x = self.relu2(x)
        x = self.max_pool(x)
        # calculate how many output units there will be after the convolutional layers
        if self.to_linear is None:
            self.to_linear = x.shape[1] * x.shape[2] * x.shape[3]
        return x

    def forward(self, x, tSNE_list=False):
        x = self.convs(x)
        x = x.view(-1, self.to_linear)
        if tSNE_list:
            feature_vectors = x
        x = self.linear(x)
        if tSNE_list:
            return x, feature_vectors
        return x

    def forward_layer_visualization(self, x):
        x = self.conv_layer_1(x)
        plot_hidden_layers(x, 6, "After 1st convolutional layer")
        x = self.batch_norm_1(x)
        x = self.relu1(x)
        plot_hidden_layers(x, 6, "After 1st relu operation")
        x = self.conv_layer_2(x)
        plot_hidden_layers(x, 10, "After 2nd convolutional layer")
        x = self.batch_norm_2(x)
        x = self.relu2(x)
        plot_hidden_layers(x, 10, "After 2nd relu operation")
        x = self.max_pool(x)
        plot_hidden_layers(x, 10, "After max-pool layer")


# specifies the linear layers with the input being calculated before (to_linear)
def sequential_layers_linear(to_linear):
    return nn.Sequential(nn.Linear(to_linear, 10))


def train_cnn(x, y, epochs=20, learningRate=0.007, l2_weight_decay=0.0012, batch_size=20):
    x, y = matrices_to_tensors(x, y)
    model = Net()
    model = model.float()
    optimizer = Adam(model.parameters(), lr=learningRate, weight_decay=l2_weight_decay)
    criterion = nn.CrossEntropyLoss()
    loss_list = []
    # batch sizes are calculated and sorted
    if not batch_size or batch_size > x.shape[0]:
        batch_size = x.shape[0]
    batch_num = x.shape[0] / batch_size
    x = x.reshape(-1, batch_size, 1, 16, 15)
    y = y.reshape(-1, batch_size)

    for epoch in range(0, epochs):
        # loop over the number of batches feeds in batch_size many images and performs backprop
        # then again and so on
        for i in range(0, int(batch_num)):
            # here we feed in training data and perform backprop according to the loss
            # run the forward pass
            outputs = model.forward(x[i])
            y = y.long()
            loss = criterion(outputs, y[i])
            loss_list.append(loss.item())

            # backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model, loss_list


def print_layers(model, x, y):
    # x, _ = matrices_to_tensors(x, y)
    x = x.reshape(1, 1, 16, 15)
    x = torch.from_numpy(x)
    x = x.float()
    print(y)
    output = model.forward_layer_visualization(x)


def eval_cnn(model, x, y, tSNE_list=False):
    x, y = matrices_to_tensors(x, y)
    if tSNE_list:
        output, feature_vectors = model.forward(x, tSNE_list=True)
    else:
        output = model.forward(x)
    total = y.size(0)
    _, predicted = torch.max(output.data, 1)
    correct = 0
    wrong_digits = []
    wrong_pred = []
    wrong_y = []
    for i in range(total):
        if predicted[i] == y[i]:
            correct += 1
        else:
            wrong_digits.append(x[i][0].tolist())
            wrong_pred.append(predicted[i].item())
            wrong_y.append(y[i].item())
    if tSNE_list:
        return correct / total, wrong_digits, wrong_pred, wrong_y, feature_vectors
    return correct / total, wrong_digits, wrong_pred, wrong_y


def crossvalidationCNN(x, y, k):
    # setup the k-fold split
    total = x.shape[0]
    k = split_check(total, k)
    bin_size = int(total / k)
    folds_x = np.array(np.array_split(x, k))
    folds_y = np.array(np.split(y, k))
    acc_train_m = list()
    acc_test_m = list()
    m_list = list()

    # define m range in this case m corresponds with epochs
    # to change what is going to vary with m, mention in the train_cnn function
    # eg. learning_rate = m, batch_size = m ...
    # also declare what you change for the graph legend
    # type 'architecture' if changing architecture, make there only be 1 step 
    change = 'l2 regularization'
    start = 0
    stop = 1
    step = 0.01

    # new folder for each new run, except if ran size is 1
    # file with list of ave accuracies
    # plot 
    num = get_run_number('data/numberOfOptimisations.txt')
    newfile = f'data/optimisations/opt_{num}'
    os.makedirs(newfile, exist_ok=True)
    best_m = 0
    best_m_acc = 0
    m_range = np.arange(start, stop, step)
    print(f'training and evaluating {k * len(m_range)} models')

    for m in tqdm(m_range, desc='m values', position=0):  # loop over given m settings
        # mFile = f'{newfile}/{change}_{m}'
        # os.makedirs(mFile, exist_ok= True)
        acc_train = list()
        acc_test = list()
        for fold in tqdm(range(0, k), desc='folds', position=1,
                         leave=False):  # train a new model for each fold and for each m
            train_x, train_y, test_x, test_y = get_fold(folds_x, folds_y, fold)
            model, loss = train_cnn(train_x, train_y, l2_weight_decay=m, epochs=16, batch_size=30)
            acc, _, _, _ = eval_cnn(model, train_x, train_y)
            acc_train.append(acc)
            acc, _, _, _ = eval_cnn(model, test_x, test_y)
            acc_test.append(acc)
        mean_train_acc = round(np.mean(acc_train), 4)
        mean_test_acc = round(np.mean(acc_test), 4)
        acc_train_m.append(mean_train_acc)
        acc_test_m.append(mean_test_acc)
        if mean_test_acc > best_m_acc:
            best_m_acc = mean_test_acc
            best_m = m
        m_list.append(round(m, 4))
    save_best_m(newfile, change, round(best_m, 4), round(best_m_acc, 4))
    save_accuracies_sum(newfile, m_list, acc_train_m, acc_test_m)
    return acc_train_m, acc_test_m, m_list, change, newfile


def test_model(x_train, y_train, x_test, y_test):
    num = get_run_number('data/numberOfRuns.txt')
    new_file = f"data/test_runs/test_run_{num}"
    plots_file = f'{new_file}/error_plots'
    os.makedirs(new_file, exist_ok=True)
    os.makedirs(plots_file, exist_ok=True)
    # directories for saving model summary, plots and accuracies
    # train and test several models for average testing accuracy
    x_train, x_test = vectors_to_matrices(x_train), vectors_to_matrices(x_test)
    accuracy = []
    for i in tqdm(range(10)):
        model, loss = train_cnn(x_train, y_train)
        acc_test, wrong_x, wrong_predicted, wrong_y = eval_cnn(model, x_test, y_test)
        accuracy.append(acc_test)
        plotWrongDigits(wrong_x, wrong_predicted, wrong_y, plots_file, i)
    print(np.mean(accuracy))
    save_model(new_file, model, sum(accuracy) / len(accuracy))
    save_accuracies(new_file, accuracy)
