import numpy as np
import pandas as pd
from scipy.ndimage import interpolation
from sklearn.model_selection import train_test_split


def moments(image):
    ''' Function to first calculate moments of the image data, which is the first step to deskewing the image. '''
    c0, c1 = np.mgrid[:image.shape[0], :image.shape[1]]
    totalImage = np.sum(image)
    m0 = np.sum(c0*image)/totalImage
    m1 = np.sum(c1*image)/totalImage
    m00 = np.sum((c0-m0)**2*image)/totalImage
    m11 = np.sum((c1-m1)**2*image)/totalImage
    m01 = np.sum((c0-m0)*(c1-m1)*image)/totalImage
    mu_vector = np.array([m0, m1])
    covariance_matrix = np.array([[m00, m01], [m01, m11]])
    return mu_vector, covariance_matrix


def deskew(image):
    ''' Function used for deskewing the image which internally first calls the moment function described above. '''
    c, v = moments(image)
    alpha = v[0, 1]/v[0, 0]
    affine = np.array([[1, 0], [alpha, 1]])
    ocenter = np.array(image.shape)/2.0
    offset = c-np.dot(affine, ocenter)
    return interpolation.affine_transform(image, affine, offset=offset)


def scale(vect):
    return (vect-vect.min())/(vect.max()-vect.min())


def split_data():
    df_train = pd.read_csv("data/dataset.csv")
    df_X = df_train.drop("label", axis=1)
    df_X = df_X.apply(lambda x: deskew(
        x.values.reshape(28, 28)).flatten(), axis=1)
    X = df_X.apply(scale)
    X = X.dropna(how='all')
    y = df_train["label"]
    return train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)
