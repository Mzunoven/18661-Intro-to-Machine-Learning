#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import argparse
import math

from operator import itemgetter


class Model(object):
    """
     Ridge Regression(Linear Least Squares Regression with Tikhonov regularization).
    """

    def fit(self, X, y, alpha=0):
        """
        Fits the ridge regression model to the training data.

        Arguments
        ----------
        X: nxp matrix of n examples with p independent variables
        y: response variable vector for n examples
        alpha: regularization parameter.
        """
        # Your code here
        I = np.eye(X.shape[1])
        self.weight = np.linalg.inv((X.T).dot(X)+alpha*I).dot(X.T).dot(y)

    def predict(self, X):
        """
        Predicts the dependent variable of new data using the model.

        Arguments
        ----------
        X: nxp matrix of n examples with p covariates

        Returns
        ----------
        response variable vector for n examples
        """
        # Your code here
        return X.dot(self.weight)

    def validate(self, X, y):
        """
        Returns the RMSE(Root Mean Squared Error) when the model is validated.

        Arguments
        ----------
        X: nxp matrix of n examples with p covariates
        y: response variable vector for n examples

        Returns
        ----------
        RMSE when model is used to predict y
        """
        # Your code here
        self.rss = ((self.weight.T).dot(X.T).dot(X).dot(
            self.weight))-2*(((X.T).dot(y)).T).dot(self.weight)
        return self.rss

# run command:
# python ridge.py --X_train_set=Xtraining.csv --y_train_set=Ytraining.csv --X_val_set=Xvalidation.csv --y_val_set=Yvalidation.csv --y_test_set=Ytesting.csv --X_test_set=Xtesting.csv


if __name__ == '__main__':

    # Read command line arguments

    parser = argparse.ArgumentParser(
        description='Fit a Ridge Regression Model')
    parser.add_argument('--X_train_set', required=True,
                        help='The file which contains the covariates of the training dataset.')
    parser.add_argument('--y_train_set', required=True,
                        help='The file which contains the response of the training dataset.')
    parser.add_argument('--X_val_set', required=True,
                        help='The file which contains the covariates of the validation dataset.')
    parser.add_argument('--y_val_set', required=True,
                        help='The file which contains the response of the validation dataset.')
    parser.add_argument('--X_test_set', required=True,
                        help='The file which containts the covariates of the testing dataset.')
    parser.add_argument('--y_test_set', required=True,
                        help='The file which containts the response of the testing dataset.')

    args = parser.parse_args()

    # Parse training dataset
    X_train = np.genfromtxt('Xtraining.csv', delimiter=',')
    y_train = np.genfromtxt('Ytraining.csv', delimiter=',')

    # Parse validation set
    X_val = np.genfromtxt('Xvalidation.csv', delimiter=',')
    y_val = np.genfromtxt('Yvalidation.csv', delimiter=',')

    # Parse testing set
    X_test = np.genfromtxt('Xtesting.csv', delimiter=',')
    y_test = np.genfromtxt('Ytesting.csv', delimiter=',')

    # find the best regularization parameter
    # Your code here

    # plot rmse versus lambda
    # Your code here

    # plot predicted versus real value
    # Your code here

    # plot regression coefficients
    # Your code here
