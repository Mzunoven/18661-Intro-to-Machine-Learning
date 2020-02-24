import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Load dataset
dataset = np.loadtxt('/Users/muzo01/Cpp_projects/gmm_data.txt')
# print(dataset.shape)
X1 = dataset[:, 0:2]
X2 = dataset[:, 2:4]
X3 = dataset[:, 3:]
# print(X3)

# Question3.(a): GMM
gmm = GaussianMixture(
    n_components=3, covariance_type='spherical', init_params='kmeans').fit(dataset)
miu = gmm.means_
sigma = gmm.covariances_
print(miu)

# Question3.(b):
gmm1 = GaussianMixture(
    n_components=3, covariance_type='spherical', init_params='random').fit(X1)
gmm2 = GaussianMixture(
    n_components=3, covariance_type='spherical', init_params='random').fit(X2)
gmm3 = GaussianMixture(
    n_components=3, covariance_type='spherical', init_params='random').fit(X3)
label1 = gmm1.predict(X1)
label2 = gmm2.predict(X2)
label3 = gmm3.predict(X3)
# print(label3)
colors = ['b', 'g', 'r']
plt.figure()
plt.subplot(221)
plt.scatter(X1[:, 0], X1[:, 1], c=label1, cmap='brg_r', s=5)
plt.subplot(222)
plt.scatter(X2[:, 0], X2[:, 1], c=label2, cmap='brg_r', s=5)
plt.subplot(223)
plt.scatter(X3[:, 0], X3[:, 1], c=label3, cmap='brg_r', s=5)
plt.show()
