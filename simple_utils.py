import random

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

import geppy as gep

s = 0
random.seed(s)
np.random.seed(s)


def boston_house_data():
    origin_data = load_boston()
    data = pd.DataFrame(origin_data.data, columns=origin_data.feature_names)
    data['MEDV'] = pd.Series(origin_data.target)

    # process outliers
    data = data[~(data['MEDV'] >= 50.0)]

    # select important variable
    column_name = ['LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX', 'DIS', 'AGE']
    x = data.loc[:, column_name]
    y = data['MEDV']
    # return StandardScaler().fit_transform(x.values), StandardScaler().fit_transform(y.values.reshape(-1,1)), column_name
    return x.values, y.values, column_name


threshold = 1e-6


def protect_divide(x1, x2):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x2) > threshold, np.divide(x1, x2), 1.)


def generate_primitive_set(input_name):
    pset = gep.PrimitiveSet('Main', input_names=input_name)
    pset.add_function(np.add, 2)
    pset.add_function(np.subtract, 2)
    pset.add_function(np.multiply, 2)
    pset.add_function(protect_divide, 2)
    pset.add_ephemeral_terminal(name='enc', gen=lambda: random.randint(-1, 1))
    return pset


if __name__ == '__main__':
    X, Y, input_name = boston_house_data()
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
