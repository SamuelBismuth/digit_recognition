'''
Python version: 3.9
Submitters: Yishay Seroussi 305027948, Samuel Bismuth 342533064.
'''


# https://www.kaggle.com/sanesanyo/digit-recognition-using-svm-with-98-accuracy


from sklearn.svm import SVC
import repackage
repackage.up()
from data import split_data


def SVM(train_x, test_x, train_y, test_y):
    clf = SVC(C=1, kernel="linear")
    clf.fit(train_x.tolist(), train_y.tolist())
    print('SVM accuracy: {0}'.format(clf.score(test_x.tolist(), test_y.tolist())))


if __name__ == "__main__":
    train_x, test_x, train_y, test_y = split_data()
    SVM(train_x, test_x, train_y, test_y)
