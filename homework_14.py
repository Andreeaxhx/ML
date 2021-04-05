from sklearn.datasets import make_blobs
import numpy as np

n_samples = 500
random_state = 10
X, y = make_blobs(n_samples=n_samples, centers=[(0, 0)], random_state=random_state)
# Anisotropically distributed data
transformation1 = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
transformation2 = [[0.60834549, 0.63667341], [0.40887718, 0.85253229]]
anis1 = np.dot(X, transformation1)
anis2 = np.dot(X, transformation2)
d = np.vstack([anis1, anis2])

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

km = KMeans(n_clusters=2)
clusters = km.fit_predict(d)
centroids = km.cluster_centers_
# Plot clusters
fig, ax = plt.subplots(figsize=(8, 6))
c = ListedColormap(['red', 'green'])
plt.scatter(centroids[:, 0], centroids[:, 1], s=100, marker='x', c=range(len(centroids)), cmap=c)
plt.scatter(d[:, 0], d[:, 1], c=clusters, cmap=c)

plt.title(f'Iterations: {km.n_iter_}')
plt.show()

from sklearn.mixture import GaussianMixture
from matplotlib.colors import LogNorm
from tools.plots import add_ellipses

clf = GaussianMixture(n_components=2, covariance_type='full')
clf.fit(d)

fig, ax = plt.subplots(figsize=(8, 6))
clusters = clf.predict(d)

c1 = ListedColormap(['red', 'green'])
c2 = ['lightcoral', 'lightgreen']
ax.scatter(d[:, 0], d[:, 1], c=clusters, cmap=c1)

add_ellipses(clf, ax, c2)

plt.title('GMM clusters')
plt.axis('tight')
plt.show()