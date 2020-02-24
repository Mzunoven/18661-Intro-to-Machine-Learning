import pandas as pd
import numpy as np
from sklearn import linear_model
from matplotlib import pyplot as plt


input_file = "poly_reg_data.csv"

input_data = pd.read_csv(input_file)


######################################################################
'''
This was for splitting training set, validation set, and test set manually and deterministically.
Students must have different codes for this part. 
'''
n = len(input_data['y'])
n_train = 25
n_val = n - n_train

x = input_data['x']
x_train = np.matrix(x[:n_train]).T
x_val = np.matrix(x[n_train:]).T

y = input_data['y']
y_train = np.matrix(y[:n_train]).T
y_val = np.matrix(y[n_train:]).T

############################################################################


def PolyCoefficients(x, coeffs):
    """ Returns a polynomial for ``x`` values for the ``coeffs`` provided.

    The coefficients must be in ascending order (``x**0`` to ``x**o``).
    """
    o = len(coeffs)
    y = 0
    for i in range(o):
        y += coeffs[i]*x**i
    return np.transpose(y)


def fit(X, y, k):
    phi = np.ones((n_train, 1))
    for i in range(k):
        phi = np.concatenate([phi, np.power(X, i+1)], axis=1)
    phi = np.matrix(phi)
    w = np.linalg.inv(phi.T*phi)*phi.T*y
    return w


def validate(X, y, w, num_points):
    k = len(w)-1
    phi = np.ones((num_points, 1))
    for i in range(k):
        phi = np.concatenate([phi, np.power(X, i+1)], axis=1)
    phi = np.matrix(phi)
    err = np.linalg.norm(y - phi*w)
    return err


errs = np.zeros((10, 1))
train_errs = np.zeros((10, 1))
f, axes = plt.subplots(1, 4, sharey=True)
subplt_idx = 0

for k in range(1, 11):
    w = fit(x_train, y_train, k)
    errs[k-1] = validate(x_val, y_val, w, n_val)
    train_errs[k-1] = validate(x_train, y_train, w, n_train)
    if k in [1, 3, 5, 10]:
        plt.figure(1)
        phi = np.ones((n_val, 1))
        for i in range(k):
            phi = np.concatenate([phi, np.power(x_val, i+1)], axis=1)
        phi = np.matrix(phi)
        y_hat = phi*w
        for i in range(n_val):
            axes[subplt_idx].scatter(x_val[i, 0], y_val[i, 0], color='blue')
        x = np.linspace(0, 5)
        axes[subplt_idx].plot(x, PolyCoefficients(x, w))
        axes[subplt_idx].set_title('k='+str(k))
        subplt_idx += 1


plt.figure(2)
plt.plot(list(range(1, 11)), errs)

plt.figure(3)
plt.plot(list(range(1, 11)), train_errs)

plt.savefig()
