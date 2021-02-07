'''
Python version: 3.9
Submitters: Yishay Seroussi 305027948, Samuel Bismuth 342533064.
'''


# https://www.kaggle.com/nikhilmudholkar/digit-recognition-using-knn
# https://towardsdatascience.com/building-a-k-nearest-neighbors-k-nn-model-with-scikit-learn-51209555453a


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import repackage
repackage.up()
from data import split_data


def best_param(train_x, train_y):
    knn2 = KNeighborsClassifier()
    param_grid = {'n_neighbors': np.arange(1, 20)}
    knn_gscv = GridSearchCV(knn2, param_grid, cv=5)
    knn_gscv.fit(train_x.tolist(), train_y.tolist())
    print(knn_gscv.best_params_)


def KNN(train_x, test_x, train_y, test_y):
    k = 2  # Arbitrary.
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(train_x.tolist(), train_y.tolist())
    print('KNN accuracy: {0}'.format(knn.score(test_x.tolist(), test_y.tolist())))


if __name__ == "__main__":
    train_x, test_x, train_y, test_y = split_data()
    best_param(train_x, train_y)
    KNN(train_x, test_x, train_y, test_y)