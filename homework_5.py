#Exercise 1 -- Manolache Gabriel
#Given the following dataset with two input random variables  𝑋1  and  𝑋2  and a target variable  𝑌 , we want to compare two extreme decision tree algorithms:

#OVERFIT will build a full standard ID3 decision tree, with no pruning;
#UNDERFIT will not split the intervals at all, always having a single node (which is both root and decision).
#1.Plot the full OVERFIT tree.
#2.What is the CVLOO error for OVERFIT?
#3.What is the CVLOO error for UNDERFIT?

import pandas as pd
d = pd.DataFrame({'X1': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8],
                  'X2': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                  'Y' : [0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]})
import matplotlib.pyplot as plt
from sklearn import tree
from statistics import mean
X= d[['X1','X2']]
Y = d['Y']
# 1
dt = tree.DecisionTreeClassifier(criterion='entropy').fit(X,Y)

fig, ax = plt.subplots(figsize=(10, 10))
f = tree.plot_tree(dt, ax=ax, fontsize=10, feature_names=X.columns)
print('1.  Full OVERFIT Tree')
plt.show()
# 2
from sklearn.model_selection import LeaveOneOut,cross_val_score
loo = LeaveOneOut()
print('2.  CVLOO error for OVERFIT: ',1-mean(cross_val_score(dt, X, Y, cv=loo)))
# 3
model=tree.DecisionTreeClassifier()
d = pd.DataFrame({'X1': [2,3],
                  'X2': [2,1],
                  'Y' : [0,0]})
X= d[['X1','X2']]
Y = d['Y']

dt = tree.DecisionTreeClassifier(criterion='entropy').fit(X,Y)
print('3. CVLOO error for UNDERFIT: ',1-mean(cross_val_score(dt, X, Y,cv=loo)))
fig, ax = plt.subplots(figsize=(1,1 ))
f = tree.plot_tree(dt, ax=ax, fontsize=10, feature_names=X.columns)
plt.show()


#Exercise 3 -- Cruceanu Cosmin
#Given the dataset below:

#1.plot the points and the labels using matplotib.pyplot.scatter;
#2.train a regular decision tree, then plot its decision surface using matplotlib.pyplot.contourf.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

d = pd.DataFrame({'X1': [1, 2, 3, 3, 3, 4, 5, 5, 5, 3],
                  'X2': [2, 3, 1, 2, 4, 4, 1, 2, 4, 3],
                  'Y': [1, 1, 0, 0, 1, 0, 1, 1, 0, 1]})
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(d['X1'].values, d['X2'].values, color='r', alpha=0.3)
# ax.scatter(d['Y'].values, d['X2'].values, color='b', alpha=0.3)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
plt.show()

x_train, x_test, y_train, y_test = train_test_split(d[['X1', 'X2']], d['Y'], test_size=0.25, random_state=1)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
classifier = DecisionTreeClassifier().fit(x_train, y_train)
y_predict = classifier.predict(x_test)

x_set, y_set = sc.inverse_transform(x_train), y_train

print(metrics.accuracy_score(y_test, y_predict))

min1, max1 = x_set[:, 0].min() - 1, x_set[:, 0].max() + 1
min2, max2 = x_set[:, 1].min() - 1, x_set[:, 1].max() + 1

x1_scale = np.arange(min1, max1, 0.1)
x2_scale = np.arange(min2, max2, 0.1)

x1, x2 = np.meshgrid(x1_scale, x2_scale)

x_flatten = np.array([x1.ravel(), x2.ravel()])
x_transformed = sc.transform(x_flatten.T)
z_prediction = classifier.predict(x_transformed).reshape(x1.shape)
plt.contour(x1, x2, z_prediction, alpha=0.75, cmap=ListedColormap(('#386cb0', '#f0027f')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], color=ListedColormap(('red', 'green'))(i), label=j)
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()


#Exercise 4 -- Eu
#Given the following dataset, with input attributes  𝐴 ,  𝐵 , and  𝐶  and target attribute  𝑌 ,
#predict the entry  𝐴=0,𝐵=0,𝐶=1  using BernoulliNB(alpha=1e-10) and predict_proba() then manually calculate the probabilities using the formulas.

import pandas as pd
from sklearn.naive_bayes import BernoulliNB

features = ['A', 'B', 'C']
target = 'Y'

messages = pd.DataFrame({'A': [0, 0, 1, 0, 1, 1, 1],
                         'B': [0, 1, 1, 0, 1, 0, 1],
                         'C': [1, 0, 0, 1, 1, 0, 0],
                         'Y': [0, 0, 0, 1, 1, 1, 1]})

new_messages = pd.DataFrame({'A':[0], 'B':[0], 'C':[1]})

X = messages[features]
y = messages[target]

cl = BernoulliNB(alpha=0.0000000001).fit(X, y)
print(cl.predict(new_messages))
print(cl.predict_proba(new_messages))

#Exercise 5 -- Samson Theodor
#Consider two random variables  𝑋1  and  𝑋2  and a label  𝑌  assigned to each instance as in the followind dataset.

#1.Classify the instance  𝑋1=0,𝑋2=0  using Naive Bayes.
#2.According to Naive Bayes, what is the probability that this classification is correct?
#3.How many probabilities are estimated by the model (check the class_log_prior_ and feature_log_prob_ attributes)?
#4.How many probabilities would be estimated by the model if there were  𝑛  features instead of 2?

import pandas as pd
#from tools.pd_helpers import apply_counts

# d = pd.DataFrame({'X1': [0, 0, 1, 1, 0, 0, 1, 1],
#                   'X2': [0, 0, 0, 0, 1, 1, 1, 1],
#                   'C' : [2, 18, 4, 1, 4, 1, 2, 18],
#                   'Y' : [0, 1, 0, 1, 0, 1, 0, 1]})
# d=apply_counts(d, 'C')

features = ['X1', 'X2']
target = ['Y']
print("Afisez setul de date")
d=pd.DataFrame(
    [(0, 0, 0),
     (0, 0, 1),
     (1, 0, 0),
     (1, 0, 1),
     (0, 1, 0),
     (0, 1, 1),
     (1, 1, 0),
     (1, 1, 1)],
    columns=features+target)

print(d)

new_d=pd.DataFrame(
    [(0, 0)],
    columns=features
)

print("Afisez datele pentru care vreau sa fac clasificarea")
print(new_d)

from sklearn.naive_bayes import BernoulliNB
X = d[features]
Y = d[target]
cl = BernoulliNB().fit(X, Y.values.ravel())
print("Pentru X1=0 si X2=0 modelul prezice : ",cl.predict(new_d))
#1.rezultatul clasificarii
print("Modelul prezice sansele urmatoare : ",cl.predict_proba(new_d))
#2.sansele pt fiecare valoare din Y pt a fi aleasa
print("Pentru clasele din Y", cl.classes_)
#valorile din Y

print("Atributul class_log_prior_", cl.class_log_prior_)
print("Atributul feature_log_prob_", cl.feature_log_prob_)