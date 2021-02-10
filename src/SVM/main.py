'''
Python version: 3.9
Submitters: Yishay Seroussi 305027948, Samuel Bismuth 342533064.
'''


# https://www.kaggle.com/sanesanyo/digit-recognition-using-svm-with-98-accuracy


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import repackage
repackage.up()
from data import split_data


def best_param(train_x, train_y):
    svc = SVC()
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['linear', 'poly']
    }
    grid = GridSearchCV(svc, param_grid, refit=False, verbose=2)
    grid.fit(train_x.to_list(), train_y.tolist())
    print(grid.best_params_)


def SVM(train_x, test_x, train_y, test_y):
    clf = SVC(C=0.1, gamma=0.1, kernel="poly")
    clf.fit(train_x.tolist(), train_y.tolist())
    print('SVM accuracy: {0}'.format(
        clf.score(test_x.tolist(), test_y.tolist())))


if __name__ == "__main__":
    train_x, test_x, train_y, test_y = split_data()
    # {'C': 0.1, 'gamma': 0.1, 'kernel': 'poly'}
    # best_param(train_x, train_y)
    SVM(train_x, test_x, train_y, test_y)
