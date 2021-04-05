#Exercise 1 -- Corina Iftinca
#Ternary classification

#The following code creates a small dataset with two attributes and a target variable with three possible values.
# 1.Calculate the information gain for X1 and X2 relative to Y.
# 2.Based on these calculations, what attribute will be used for the first node of the ID3 tree?
# 3.Learn the entire tree and classify the instance {'X1': 0, 'X2': 1}.

import math
import pandas as pd
X = pd.DataFrame({'X1': [1, 1, 1, 1, 0, 0],
                  'X2': [1, 1, 1, 0, 0, 0]})
Y = pd.Series([1, 1, 2, 3, 2, 3])

HY=-3*(2/6*math.log2(2/6))

HX1equals1=-2/4*math.log2(2/4)-2*1/4*math.log2(1/4)
HX1equals0=0-2*1/2*math.log2(1/2)
HX1cond=4/6*HX1equals1+2/6*HX1equals0
IGX1=HY-HX1cond

HX2equals1=-2/3*math.log2(2/3)-1/3*math.log2(1/3)
HX2equals0=0-1/3*math.log2(1/3)-2/3*math.log2(2/3)
HX2cond=3/6*HX2equals1+3/6*HX2equals0
IGX2=HY-HX2cond

print("Y entropy: ", HY)
print("X1 conditional entropy: ", HX1cond)
print("X2 conditional entropy: ", HX2cond)
print("Information Gain for X1", IGX1)
print("Information Gain for X2", IGX2)

if IGX1>IGX2:
    print("X1 is the first node of the ID3 tree")
else:
    print("X2 is the first node of the ID3 tree")


#Exercise 2 -- Draghici Constantin
#ID3 as a "greedy" algorithm

#The following code creates a dataset with features A, B, C and target variable Y.

#1.Find the decision tree using ID3. Is it consistent with the training data (does it have 100% accuracy)?
#2.Is there a less deep decision tree consistent with the above data? If so, what logic concept does it represent?

import numpy as np
import pandas as pd
eps = np.finfo(float).eps
import pprint

X = pd.DataFrame({'A': [1, 1, 0, 0],
                  'B': [1, 0, 1, 0],
                  'C': [0, 1, 1, 1]})
Y = pd.Series([0, 1, 1, 0])
X['Y'] = Y
def find_entropy(X):   #calculez entropia generala
    Y = X.keys()[-1]
    entropy = 0
    values = X[Y].unique()
    for value in values:
        fraction = X[Y].value_counts()[value] / len(X[Y])
        entropy += -fraction * np.log2(fraction)
    return entropy
def find_entropy_attribute(X, attribute):   #calculez entropia unui atribut
    Y = X.keys()[-1]
    target_variables = X[Y].unique()
    variables = X[attribute].unique()
    entropy2 = 0
    for variable in variables:
        entropy = 0
        for target_variable in target_variables:
            num = len(X[attribute][X[attribute] == variable][X[Y] == target_variable])
            den = len(X[attribute][X[attribute] == variable])
            fraction = num / den
            entropy += -fraction * np.log2(fraction + eps)  #calculez entropia pentru fiecare valoare
        fraction2 = den / len(X)
        entropy2 += -fraction2 * entropy  #calculez entropia pentru atibut
    return abs(entropy2)
def find_winner(X):
    IG = []
    for key in X.keys()[:-1]:
        IG.append(find_entropy(X) - find_entropy_attribute(X, key))  #calculez information gain pentru fiecare atribut
    return X.keys()[:-1][np.argmax(IG)]   #gasesc atriutul cu cel mai mare information gain
def get_subtable(X, node, value):
    return X[X[node] == value].reset_index(drop=True)
def buildTree(X, tree=None):
    Y = X.keys()[-1]
    node = find_winner(X)
    attValue = np.unique(X[node])
    if tree is None: #creez radacina
        tree = {}
        tree[node] = {}
    for value in attValue:

        subtable = get_subtable(X, node, value) #creez un nou dataset pe care voi apela din nou finctia buildTree
        clValue, counts = np.unique(subtable['Y'], return_counts=True)

        if len(counts) == 1:
            tree[node][value] = clValue[0]
        else:
            tree[node][value] = buildTree(subtable)  # Apel recursiv
    return tree
t=buildTree(X)
pprint.pprint(t)


#Exercise 3 -- Marinoiu
#Titanic dataset

#The table bellow shows a few statistics on the survivors of the Titanic.

#1.We want to build a decision tree to predict the target variable Y (survived) based on variables C (class), G (gender) and A (age).
#Using information gain, determine which of the three variables will be used in the root node.
#2.What is the training accuracy of the decision tree consisting only of the root node above?
#3.If you were to build the full tree using all attributes, what would be the training accuracy? Note that you don’t have to actually build the full tree!


import pandas as pd
from scipy.stats import entropy
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn import metrics


def attribute_information_gain(base_entropy, attribute_name, attribute_values):
    attribute_positive = titanic[titanic[attribute_name] == attribute_values[0]]
    attribute_negative = titanic[titanic[attribute_name] == attribute_values[1]]
    count_positive_survivors = sum(attribute_positive["Survivors"])
    count_positive_deaths = sum(attribute_positive["Passengers"]) - sum(attribute_positive["Survivors"])
    count_negative_survivors = sum(attribute_negative["Survivors"])
    count_negative_deaths = sum(attribute_negative["Passengers"]) - sum(attribute_negative["Survivors"])
    return base_entropy - (
            sum(attribute_positive["Passengers"]) /
            sum(titanic["Passengers"]) *
            entropy([count_positive_survivors, count_positive_deaths], base=2) +

            sum(attribute_negative["Passengers"]) /
            sum(titanic["Passengers"]) *
            entropy([count_negative_survivors, count_negative_deaths], base=2)
    )

def sub1():
    attributes = ["Class", "Gender", "Age"]
    ig_list = [
        attribute_information_gain(base_entropy, "Class", ["Upper", "Lower"]),
        attribute_information_gain(base_entropy, "Gender", ["Male", "Female"]),
        attribute_information_gain(base_entropy, "Age", ["Adult", "Child"])
    ]
    return attributes[ig_list.index(max(ig_list))]

def sub2(attribute_name):
    attribute_values_dict = {
        "Class": ["Upper", "Lower"],
        "Gender": ["Male", "Female"],
        "Age": ["Adult", "Child"]
    }
    attribute_positive = titanic[titanic[attribute_name] == attribute_values_dict[attribute_name][0]]
    attribute_negative = titanic[titanic[attribute_name] == attribute_values_dict[attribute_name][1]]

    return((sum(attribute_positive["Survivors"]) + sum(attribute_negative["Survivors"])) /
           (sum(attribute_positive["Passengers"]) + sum(attribute_negative["Passengers"])))


def sub3(titanic2):
    titanic2["Class"] = [1 if element == "Upper" else 0 for element in titanic["Class"]]
    titanic2["Gender"] = [1 if element == "Male" else 0 for element in titanic["Gender"]]
    titanic2["Age"] = [1 if element == "Adult" else 0 for element in titanic["Age"]]
    new_list = []
    for index, line in titanic2.iterrows():
        survivors_count = line[4]
        deaths_count = line[3] - survivors_count
        for _ in range(survivors_count):
            new_list.append((line[0], line[1], line[2], 1))
        for _ in range(deaths_count):
            new_list.append((line[0], line[1], line[2], 0))
    titanic2 = pd.DataFrame(new_list,
                            columns=["Class", "Gender", "Age", "Survivor"])
    dt = tree.DecisionTreeClassifier(criterion='entropy').fit(titanic2[["Gender"]], titanic2["Survivor"])
    predicted = dt.predict(titanic2[["Gender"]])
    print("Ex. 3: Accuracy:", metrics.accuracy_score(titanic2["Survivor"], predicted))
    fig, ax = plt.subplots(figsize=(16, 6))
    f = tree.plot_tree(dt, ax=ax, fontsize=8, feature_names=titanic2[["Gender"]].columns)
    plt.show()

if __name__ == '__main__':
    titanic = pd.DataFrame([
        ('Upper', 'Male', 'Child', 5, 5),
        ('Upper', 'Male', 'Adult', 175, 57),
        ('Upper', 'Female', 'Child', 1, 1),
        ('Upper', 'Female', 'Adult', 144, 140),
        ('Lower', 'Male', 'Child', 59, 24),
        ('Lower', 'Male', 'Adult', 1492, 281),
        ('Lower', 'Female', 'Child', 44, 27),
        ('Lower', 'Female', 'Adult', 281, 176)
    ],
        columns=['Class', 'Gender', 'Age', 'Passengers', 'Survivors'])
    count_survivors = sum(titanic["Survivors"])
    count_deaths = sum(titanic["Passengers"]) - sum(titanic["Survivors"])
    base_entropy = entropy([count_survivors, count_deaths], base=2)
    attribute = sub1()
    print("Ex. 1:", attribute)
    print("Ex. 2:", 1-sub2(attribute))
    sub3(titanic)

#Exercise 4 -- Ruxandra Axinia
#Exoplanets, one-hot encoding

#Given a dataset with data regarding 800 exoplanets, fit a decision tree to find how well Size and Orbit describe if a planet is habitable.
#In other words, find the training accuracy of a decision tree model that uses those two variables to predict Habitable and also print the resulting tree.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
from tools.pd_helpers import apply_counts


exoplanets = pd.DataFrame([
  ('Big', 'Near', 'Yes', 20),
  ('Big', 'Far', 'Yes', 170),
  ('Small', 'Near', 'Yes', 139),
  ('Small', 'Far', 'Yes', 45),
  ('Big', 'Near', 'No', 130),
  ('Big', 'Far', 'No', 30),
  ('Small', 'Near', 'No', 11),
  ('Small', 'Far', 'No', 255)
],
columns=['Big', 'Orbit', 'Habitable', 'Count'])
exoplanets = apply_counts(exoplanets, 'Count')


#ne transformam valorile in format binar
dfDummies = pd.get_dummies(exoplanets['Big'], prefix = 'Big')
dfDummies2 = pd.get_dummies(exoplanets['Orbit'], prefix = 'Orbit')
dfDummies3 = pd.get_dummies(exoplanets['Habitable'], prefix = 'Habitable')

#concatenam pentru matricea pe baza careia se face arborele
dfDummies =  pd.concat([dfDummies['Big_Big'], dfDummies2['Orbit_Far']], axis=1)
dfDummies =  pd.concat([dfDummies, dfDummies3['Habitable_Yes']], axis=1)

#cream arborele de decizie
x =  dfDummies[['Big_Big','Orbit_Far']]
y = dfDummies[['Habitable_Yes']]
dt = tree.DecisionTreeClassifier(criterion='entropy').fit(x,y)

#se printeaza arborele de decizie gasit
fig, ax = plt.subplots(figsize=(7, 8))
f = tree.plot_tree(dt, ax=ax, fontsize=10, feature_names=['Big','Orbit','Habitable'])

#se dau noile instante
new_instances = pd.DataFrame([
    (0, 1),
    (1, 1),
    (0, 0),
    (1, 0)
],
columns=['Big (is_big)', 'Orbit (is_near)' ])

#cu ajutorul functiei predict, se va estima campul Habitable
new_with_pred = new_instances.copy()
new_with_pred['Predicted_Habitable'] = dt.predict(new_instances)

#se calculeaza acuratetea
print(" Acuratetea este de:  %.2f%%  " % (dt.score(x, y)*100))

new_with_pred

#Exercise 5 -- Cirloanta
#Exoplanets, continuous variable

#Given a dataset with 9 exoplanets for which we know the Temperature as well as the target variable Habitable.

#1.Find the training accuracy of a decision tree that predicts Habitable using Temperature and print the resulting tree.
#2.Independently calculate the split points that the algorithm will use for Temperature and check it against the generated tree. (The solution does not need to be general, can be "hard-coded" for this dataset.)
#3.Independently calculate the entropy of the root node of the generated tree.


import pandas as pd
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import math

def H(p):
    def log_zero(x): return 0 if x is 0 else math.log2(x)
    return -sum(i*log_zero(i) for i in p)


exoplanets = pd.DataFrame([
    ('Big', 'Near', 205, 0),
    ('Big', 'Far', 205, 0),
    ('Small', 'Near', 260, 1),
    ('Small', 'Far', 380, 1),
    ('Big', 'Near', 205, 0),
    ('Big', 'Far', 260, 1),
    ('Small', 'Near', 260, 1),
    ('Small', 'Far', 380, 0),
    ('Small', 'Far', 380, 0)
],
    columns=['Big', 'Orbit', 'Temperature', 'Habitable'])

exo = pd.DataFrame(exoplanets[['Temperature', 'Habitable']])

input_exo = pd.DataFrame(exo['Temperature'])

dt = tree.DecisionTreeClassifier(criterion='entropy').fit(
    exo[['Temperature']], exo['Habitable'])
fig, ax = plt.subplots(figsize=(7, 8))
f = tree.plot_tree(dt, ax=ax, fontsize=10, feature_names=exo.columns)

pred_exo = input_exo.copy()
pred_exo['Habitable'] = dt.predict(input_exo)
# print(exo)
print("Decision Tree accuracy = {}".format(
    dt.score(input_exo, exo['Habitable'])))

entropy = -(4.0/9) * math.log2(4.0/9) - (5.0/9) * math.log2(5.0/9)

print("Root node entropy = H[4+, 5-] = {}".format(entropy))
plt.show()