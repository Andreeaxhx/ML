#Exercise 1 -- Iftinca Corina
#Learned probabilities

#Given the following run of the Naive Bayes algorithm without smoothing.

#1.Write a function that independently calculates the value of the class_log_prior_ attribute without smoothing
# using only messages as parameter. (These are the natural logarithms of class probabilities from the formula
# ∏𝑖𝑃(𝑎𝑖|𝑣𝑗)𝑃(𝑣𝑗) ).
#2.Write a function that independently calculates the value of the feature_log_prob_ attribute without smoothing
# using only messages as parameter. (These are the natural logarithms of attribute probabilities from the formula above).

import pandas as pd
from sklearn.naive_bayes import BernoulliNB
import math

# Create the training set
features = ['study', 'free', 'money']
target = 'is_spam'
messages = pd.DataFrame(
    [(1, 0, 0, 0),
     (0, 0, 1, 0),
     (1, 0, 0, 0),
     (1, 1, 0, 0)] +
    [(0, 1, 0, 1)] * 4 +
    [(0, 1, 1, 1)] * 4,
    columns=features + [target])

# Find the probability of a message to be or not spam
def log_prior_class(messages):
    count0 = 0
    count1 = 0
    lista = []
    for i in messages[target]:
        if i == 0:
            count0 += 1
        else:
            count1 += 1
    total = count0 + count1
    prob_is = count1 / total
    prob_not = count0 / total

    lista.append(math.log(prob_not))
    lista.append(math.log(prob_is))
    print(lista)


log_prior_class(messages)


# Find the probability for a message that contains one of the words 'study','free' or 'money' to be spam or not
def log_features(messages):
    lista_is = []
    lista_not = []
    lista = []
    nr_rand = len(messages[target])
    study_1 = 0
    free_1 = 0
    money_1 = 0
    study_s = 0
    free_s = 0
    money_s = 0
    spam_0 = 0
    spam_1 = 0
    for j in messages[target]:
        if j == 0:
            spam_0 += 1
        else:
            spam_1 += 1
    for i in range(nr_rand):
        if messages['study'][i] == 1 and messages[target][i] == 0:
            study_1 += 1

        if messages['free'][i] == 1 and messages[target][i] == 0:
            free_1 += 1

        if messages['money'][i] == 1 and messages[target][i] == 0:
            money_1 += 1

        if messages['study'][i] == 1 and messages[target][i] == 1:
            study_s += 1

        if messages['free'][i] == 1 and messages[target][i] == 1:
            free_s += 1

        if messages['money'][i] == 1 and messages[target][i] == 1:
            money_s += 1

            # When the probability=1,for cl.feature_log_prob_, they aproximate the probability with the number that is really close to 1
    if money_s == spam_1 or free_s == spam_1 or study_s == spam_1 or money_1 == spam_0 or free_1 == spam_0 or study_1 == spam_0:
        p_cond_5 = 0.9999999999875002
    # When the probability=0,for cl.feature_log_prob_, they aproximate the probability with the number that is really close to 0,
    # as you can calculate ln0
    if money_s == 0 or free_s == 0 or study_s == 0 or money_1 == 0 or free_1 == 0 or study_1 == 0:
        p_cond_4 = 0.0000000000125
    p_cond_not1 = study_1 / spam_0
    p_cond_not2 = free_1 / spam_0
    p_cond_not3 = money_1 / spam_0
    p_cond_6 = money_s / spam_1
    lista_not.append(math.log(p_cond_not1))
    lista_not.append(math.log(p_cond_not2))
    lista_not.append(math.log(p_cond_not3))
    lista_is.append(math.log(p_cond_4))
    lista_is.append(math.log(p_cond_5))
    lista_is.append(math.log(p_cond_6))
    lista.append(lista_not)
    lista.append(lista_is)
    print(lista)


log_features(messages)

# Create the prediction set
X = messages[features]
y = messages[target]
cl = BernoulliNB(alpha=1e-10).fit(X, y)

print(cl.class_log_prior_)
print(cl.feature_log_prob_)


#Exercise 2 -- Craciun
#Expected error rate in training

#Consider a binary classification problem with features  𝑋1  and  𝑋2  and label  𝑌 . The two features are assumed
#to be conditionally independent with respect to  𝑌  . The prior probabilities  𝑃(𝑌=0)  and  𝑃(𝑌=1)  are both equal to 0.5.
#The conditional probabilities are:

#1.Generate a DataFrame with 1000 entries and three columns ['x1', 'x2', 'y'], according to the description above,
#using the bernoulli.rvs function from scipy.
#2.After training on the DataFrame above, predict every combination of values for  𝑋1  and  𝑋2 .
#3.Calculate the average error rate on the training dataset.
#4.Create a new attribute  𝑋3  as a copy of  𝑋2 . What is the new average error rate on the training dataset?

from scipy.stats import bernoulli
import pandas as pd

prob1 = pd.DataFrame(
    {
        'y=0': [0.7, 0.3],
        'y=1': [0.2, 0.8]
    },
    index=['x1=0', 'x1=1']
)

prob2 = pd.DataFrame(
    {
        'y=0': [0.7, 0.3],
        'y=1': [0.2, 0.8]
    },
    index=['x2=0', 'x2=1']
)

print("P(X1|Y): ")
print(prob1)
print("P(X2|Y): ")
print(prob2)

# P(X1|Y) = P(X1)
# P(X2|Y) = P(X2)

def data_frame(size):
    # x1 pt y=0
    x10 = bernoulli.rvs(0.7, loc=0.3, size=size)

    # x1 pt y=1
    x11 = bernoulli.rvs(0.2, loc=0.8, size=size)

    # x2 pt y=0
    x20 = bernoulli.rvs(0.7, loc=0.3, size=size)

    # x2 pt y=1
    x21 = bernoulli.rvs(0.2, loc=0.8, size=size)

    # y
    y = bernoulli.rvs(0.5, loc=0.5, size=size)

    d = pd.DataFrame(
        {
            'x1': x10 & x11, # (x10 & y) | (x11 & ~y)
            'x2': x20 & x21, # (x20 & y) | (x21 & ~y)
            'y': y
        }
    )
    return d

DataFrame = data_frame(size=1000)
print(DataFrame)

from sklearn import tree
X1 = DataFrame[['x1']]
X2 = DataFrame['x2']
dt = tree.DecisionTreeClassifier(criterion='entropy').fit(X1, X2)
print("Prediction for every instance of X1 and X2:", dt.score(X1, X2))

from sklearn.metrics import mean_squared_error
training_error = mean_squared_error(X1, X2)
print("Training error:", training_error)

print(1-dt.score(X1, X2))

copy_of_dataframe = DataFrame.copy()
copy_of_dataframe['x3'] = DataFrame['x2']
X = copy_of_dataframe[['x1', 'x2', 'x3']]
Y = copy_of_dataframe['y']
dt = tree.DecisionTreeClassifier(criterion='entropy').fit(X, Y)
training_error = 1-dt.score(X, Y)
print("Training error after adding X3 attribute:", training_error)


#Exercise 5 -- Racovita
#Average error rate

#Given the function  𝑌=(𝐴∧𝐵)∨¬(𝐵∨𝐶)  where  𝐴 ,  𝐵  and  𝐶  are independent binary random variables, each of which having 50% chance of being 0 and 50% chance of being 1.

#Generate a DataFrame with 1000 entries and four columns A, B, C and Y, according to the description above, using the bernoulli.rvs function from scipy.
#Calculate the error rate on the training dataset.
#What is the average error rate on this training dataset for the Joint Bayes algorithm? (Note that you don't have to actually build the algorithm, just provide a theoretical justification.)

from scipy.stats import bernoulli
from sklearn.naive_bayes import BernoulliNB
import pandas as pd

size = 1000
features = ['A', 'B', 'C']
target = 'Y'
A = bernoulli.rvs(p=0.5, size=size, random_state=10)
B = bernoulli.rvs(p=0.5, size=size, random_state=20)
C = bernoulli.rvs(p=0.5, size=size, random_state=30)
d = pd.DataFrame({'A': A, 'B' : B, 'C' : C},columns=features+[target])

d['Y'] = (A & B) | (1-(B | C))

Q = d[features]
W = d[target]
cl = BernoulliNB().fit(Q, W)
print(d)
print("Accuracy on the training set")
print(cl.score(Q, W))
print("Error rate on training set")
print(1-cl.score(Q, W))

#Exercise 6 -- Delia Stoica
#Text classification

#A news company would like to automatically sort the news articles related to sport from those related to politics.
#They are using 8 key words ( 𝑤1,...,𝑤8)  and have annotated several articles in each category for training:.

#According to Naive Bayes (without smoothing), what is the probability that the document x = (1, 0, 0, 1, 1, 1, 1, 0) is about politics?

import pandas as pd
from sklearn.naive_bayes import BernoulliNB


features = [f'w{i}' for i in range(1,9)]
target = 'is_politics'

politics=pd.DataFrame([
    (1, 0, 1, 1, 1, 0, 1, 1, 1),
    (0, 0, 0, 1, 0, 0, 1, 1, 1),
    (1, 0, 0, 1, 1, 0, 1, 0, 1),
    (0, 1, 0, 0, 1, 1, 0, 1, 1),
    (0, 0, 0, 1, 1, 0, 1, 1, 1),
    (0, 0, 0, 1, 1, 0, 0, 1, 1),
    (1, 1, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 1, 0, 0, 0, 0, 0, 0),
    (1, 1, 0, 1, 0, 0, 0, 0, 0),
    (1, 1, 0, 1, 0, 0, 0, 1, 0),
    (1, 1, 0, 1, 1, 0, 0, 0, 0),
    (0, 0, 0, 1, 0, 1, 0, 0, 0),
    (1, 1, 1, 1, 1, 0, 1, 0, 0)],
    columns=features+[target])
politics

#x = (1, 0, 0, 1, 1, 1, 1, 0)

print(politics)
#print(target)

X = politics[features]
y = politics[target]
cl= BernoulliNB(alpha=1e-10).fit(X, y) #fara smoothing
#cl= BernoulliNB().fit(X, y) #cu smoothing

# Predict the message for given input
new_messages = pd.DataFrame(
  [(1, 0, 0, 1, 1, 1, 1, 0)],columns = features)
cl.predict_proba(new_messages)
print("Politics or nah: ")
print(cl.predict_proba(new_messages))