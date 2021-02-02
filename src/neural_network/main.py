'''
Python version: 3.9
Submitters: Yishay Seroussi 305027948, Samuel Bismuth 342533064.
'''


import repackage
repackage.up()
from data import split_data


def neural_network():
    print('neural_network')


if __name__ == "__main__":
    train_x, test_x, train_y, test_y = split_data()
    neural_network(train_x, test_x, train_y, test_y)