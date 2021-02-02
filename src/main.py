'''
Python version: 3.9
Submitters: Yishay Seroussi 305027948, Samuel Bismuth 342533064.
'''


from data import split_data
from k_means_clustering.main import k_means_clustering
from KNN.main import KNN
from neural_network.main import neural_network
from SVM.main import SVM


train_x, test_x, train_y, test_y = split_data()
k_means_clustering(train_x, test_x, train_y, test_y)
KNN(train_x, test_x, train_y, test_y)
neural_network(train_x, test_x, train_y, test_y)
SVM(train_x, test_x, train_y, test_y)