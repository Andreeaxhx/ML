import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import AdaBoostClassifier

from math import log, pow, e

d = pd.DataFrame({
    'X1': [ 1, 2, 2, 2.5, 3],
    'X2': [2, 1, 3, 2.5, 3],
    'Y': [1, 1,    -1, -1,   1]
})
X, Y = d[['X1', 'X2']], d['Y']

c= ['green' if l == -1 else 'red' for l in Y]

#1.Plot the dataset using pyplot.
plt.scatter(d['X1'], np.zeros_like(d['X1']), color=c)
#plt.scatter(d['X1'], d['X2'], color=c)
plt.ylabel('dataset plotted')
plt.show()

#2.Draw the decision surface corresponding to the first weak learner.
knn = AdaBoostClassifier(n_estimators=1).fit(X, Y)


#3.What are the values of  𝜖1  (training error of the first decision stump) and  𝛼1  (the "weight" of the vode of the first decision stump)?
ab = AdaBoostClassifier(n_estimators=1).fit(X, Y)
error=1-ab.score(X, Y)
print("Error --------------- ", error)

alpha=1/2*log((1-error)/error)
print("Alpha --------------- ", alpha)
print()
