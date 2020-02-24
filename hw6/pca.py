from scipy.io import loadmat
from matplotlib import pyplot as plt
import numpy as np


def pca_fun(input_data, target_d):
    # P: d x target_d matrix containing target_d eigenvectors
    # input_data is 640 * 2500, while d is set to 200
    n = input_data.shape[0]
    d = input_data.shape[1]
    # let X be the zero centered input_data
    X = input_data
    X = X - np.mean(X, axis=0)
    # convariance
    Cx = np.dot(X, X.T)
    Cx /= (1/n)
    e, u = np.linalg.eig(Cx)
    v = np.dot(X.T, u)
    P = v[:, 0:target_d]
    return P


### Data loading and plotting the image ###
data = loadmat('/Users/muzo01/Cpp_projects/face_data.mat')
image = data['image'][0]
person_id = data['personID'][0]
data_input = np.zeros((640, 2500))
# print(image[0].type)
for i in range(640):
    data_input[i, :] = image[i].reshape(1, 2500)
d = 200
P = pca_fun(data_input, d)
eigenfaces = np.zeros((5, 50, 50))
for i in range(5):
    eigenfaces[i] = P[:, i].reshape(50, 50)
fig = plt.figure()
for i in range(5):
    plt.subplot(321 + i)
    plt.imshow(eigenfaces[i], cmap='gray')
    plt.subplot(321 + i).set_title('Figure %d' % (i+1))
fig.tight_layout()
plt.show()
