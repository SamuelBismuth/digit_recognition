'''
Python version: 3.9
Submitters: Yishay Seroussi 305027948, Samuel Bismuth 342533064.
'''


# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import repackage
repackage.up()
from data import split_data


def best_param(train_x, train_y):
    mlp_gs = MLPClassifier(max_iter=100)
    parameter_space = {
        'hidden_layer_sizes': [(10, 30, 10), (20,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
    }
    clf = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=5)
    clf.fit(train_x.to_list(), train_y.tolist())
    print(clf.best_params_)


def neural_network(train_x, test_x, train_y, test_y):
    mlp = MLPClassifier()
    mlp.fit(train_x.tolist(), train_y.tolist())
    print('MLP accuracy: {0}'.format(
        mlp.score(test_x.tolist(), test_y.tolist())))


if __name__ == "__main__":
    train_x, test_x, train_y, test_y = split_data()
    best_param(train_x, train_y)
    neural_network(train_x, test_x, train_y, test_y)