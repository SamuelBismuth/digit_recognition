'''
Python version: 3.9
Submitters: Yishay Seroussi 305027948, Samuel Bismuth 342533064.
'''


# https://www.kaggle.com/nikhilmudholkar/digit-recognition-using-knn
# TODO: find something about the k param.


from sklearn.neighbors import KNeighborsClassifier
import repackage
repackage.up()
from data import split_data


def KNN(train_x, test_x, train_y, test_y):
    k = 2  # Arbitrary.
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(train_x.tolist(), train_y.tolist())
    print('KNN accuracy: {0}'.format(knn.score(test_x.tolist(), test_y.tolist())))


if __name__ == "__main__":
    train_x, test_x, train_y, test_y = split_data()
    KNN(train_x, test_x, train_y, test_y)