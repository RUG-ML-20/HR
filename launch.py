from FileIO import *
from Components import *
from Visualisation import plotNumbers, plotTrainTestPerformance, tsne_plots


# ----------loading data-------------
x_train, y_train, x_test, y_test = load(.5, plot=False)
# data_analysis(x_train, y_train, x_test, y_test)


# ---------linear regression----------
# print('linear regression scores')
# train, _, _, _ = crossval_LR(x_train, y_train, 10, False)
# test = test_LR(x_train, y_train, x_test, y_test)
# print(f'Train: {round(train[0], 4)}\nTest: {test}')
# print('finding optimal number of principal components')
# train, test, best_m, best_m_acc = crossval_LR(x_train, y_train, 10, True)
# print(f'optimial number of principal components: {best_m} with {best_m_acc}')
# plotTrainTestPerformance(train, test, 'Principal Components')
#
# print('testing final model')
# test_pca = test_LR(x_train, y_train, x_test, y_test, pca_run=True, m=best_m)
# print(f'overall model accuracy with pca: {test_pca}')


# ---------Set random parameters ------------#
# baseline model with random parameters based on literature research
# print(randomParameters())


# ------- print tSNE plots----------

# test_model(x_train, y_train, x_test, y_test)
# model, _ = train_cnn(x_train, y_train)
# acc, _, _, _ = eval_cnn(model, x_test, y_test, tSNE_list=False)
# print(acc)
# print_layers(model, x_test[5], y_test[5])
# acc, _, _, _, feature_vectors = eval_cnn(model, x_test, y_test, tSNE_list=True)
# print(acc)
# tsne_plots(x_test, y_test, feature_vectors)


# ------------cross-validation----------------

# train, test, m, change, saveLocation = crossvalidationCNN(x_train, y_train, 10)
# plotTrainTestPerformance(train, test, change, saveLocation, x_values=m)


# ---------Testing model----------

test_model(x_train, y_train, x_test, y_test)


# ----------Replot data--------------
# optfoldernum = 8
# folder = f'data/optimisations/opt_{optfoldernum}'
# m, train, test = read_results(folder)
# plotTrainTestPerformance(train, test, 'epochs', x_values=m)
