# Exercise 1 -- Andreea Padurariu

# Consider two fair dice with 6 sides each.
# 1.Print the probability distribution of the sum ( 𝑆 ) of the numbers obtained by throwing the two dice.
# 2.What is the information content in bits of the events  𝑆=2 ,  𝑆=11 ,  𝑆=5 ,  𝑆=7 .
# 3.Calculate the entropy of S.
# 4.Lets say you throw the die one at a time, and the first die shows 4.
# What is the entropy of S after this observation? Was any information gained/lost in the process of observing the outcome of the first die toss?
# If so, calculate how much information (in bits) was lost or gained.

from itertools import product
from scipy import stats
import math

A = [1, 2, 3, 4, 5, 6]
B = [1, 2, 3, 4, 5, 6]
C = list(product(A, B))

def probability(list, omega):
    return len(list)/len(omega)

def function(sum):
    E = [i for i in C if i[0] + i[1] == sum]
    return E

D=[]
for i in C:
    D.append(i[0]+i[1])

F=set(D)
probabilities=[]
print("\nProbability distribution of the sum (S):")
for i in F:
    probabilities.append(probability(function(i), D))
    print(probability(function(i), D))

def I(event):
    log_zero = lambda x: 0 if x == 0 else math.log2(x)
    return log_zero(1/probability(function(event), D))

print("\nInformation content for S=2:  ", I(2))
print("Information content for S=11: ", I(11))
print("Information content for S=5:  ", I(5))
print("Information content for S=7:  ", I(7))

def H(p):
    log_zero = lambda x: 0 if x == 0 else math.log2(x)
    return -sum(i*log_zero(i) for i in p)

print("\nEntropy of S:", H(probabilities))

no_4=len([i for i in C if i[0]==4])

def cond(sum):
    a=len([i for i in C if i[0]==4 and i[0]+i[1]==sum])
    return a

cond_probabilities=[]
for i in F:
    cond_probabilities.append(cond(i)/no_4)

print("Entropy of S, knowing first die is 4:", H(cond_probabilities))

print("\nInformation Gain: ", H(probabilities)-H(cond_probabilities))


# Exercise 3 -- Andreea Padurariu

# The following code simulates the season results for football team F.
# 1.What is the entropy of the result  𝐻(𝑟𝑒𝑠𝑢𝑙𝑡)  (ignoring all other variables)?
# 2.What are the average conditional entropies  𝐻(𝑟𝑒𝑠𝑢𝑙𝑡|𝑠𝑡𝑎𝑑𝑖𝑢𝑚)  and  𝐻(𝑟𝑒𝑠𝑢𝑙𝑡|𝑜𝑝𝑝𝑜𝑛𝑒𝑛𝑡) ?
# 3.Which of the two variables is more important in deciding the result of a game?
# Answer this question by calculating the information gain for the two variables:  𝐼𝐺(𝑟𝑒𝑠𝑢𝑙𝑡,𝑠𝑡𝑎𝑑𝑖𝑢𝑚)  and  𝐼𝐺(𝑟𝑒𝑠𝑢𝑙𝑡,𝑜𝑝𝑝𝑜𝑛𝑒𝑛𝑡) .

from itertools import product
import pandas as pd
import random
import math
random.seed(1)

def H(p):
    log_zero = lambda x: 0 if x == 0 else math.log2(x)
    return -sum(i*log_zero(i) for i in p)

opponents = ['Team '+chr(ord('A') + i) for i in range(5)]
stadiums = ['Home', 'Away']
games = pd.DataFrame(list(product(opponents, stadiums))*2, columns=['opponent', 'stadium'])
games['result'] = random.choices(["Win", "Loss", "Draw"], k=len(games))

no_win, no_draw, no_loss = 0, 0, 0

for i in games["result"]:
    if i=="Win":
        no_win+=1
    elif i=="Draw":
        no_draw+=1
    elif i=="Loss":
        no_loss+=1

p_win = no_win/len(games)
p_draw = no_draw/len(games)
p_loss = no_loss/len(games)
probabilities=[p_win, p_loss, p_draw]
H_result=H(probabilities)
print("\n----Entropy of 'result': ", H_result)

no_win_home, no_win_away = 0, 0
no_draw_home, no_draw_away = 0, 0
no_loss_home, no_loss_away = 0, 0
no_home, no_away = 0, 0
for i in range(0, len(games)):
    if games["stadium"][i]=="Home":
        no_home+=1
        if games["result"][i]=="Win":
            no_win_home+=1
        elif games["result"][i]=="Draw":
            no_draw_home+=1
        elif games["result"][i]=="Loss":
            no_loss_home+=1
    elif games["stadium"][i]=="Away":
        no_away+=1
        if games["result"][i]=="Win":
            no_win_away+=1
        elif games["result"][i]=="Draw":
            no_draw_away+=1
        elif games["result"][i]=="Loss":
            no_loss_away+=1

cond_probabilities_home=[no_win_home/no_home, no_draw_home/no_home, no_loss_home/no_home]
cond_probabilities_away=[no_win_away/no_away, no_draw_away/no_away, no_loss_away/no_away]
print("\nProbabilitatile result|home: ", cond_probabilities_home)
print("Probabilitatile result|away: ", cond_probabilities_away)

print("\nEntropy of result|home: ", H(cond_probabilities_home))
print("Entropy of result|away: ", H(cond_probabilities_away))
H_result_stadium=(no_home/len(games))*H(cond_probabilities_home)+(no_away/len(games))*H(cond_probabilities_away)
print("----Entropy of result|stadium: ", H_result_stadium)

no_win_A, no_win_B, no_win_C, no_win_D, no_win_E = 0, 0, 0, 0, 0
no_draw_A, no_draw_B, no_draw_C, no_draw_D, no_draw_E = 0, 0, 0, 0, 0
no_loss_A, no_loss_B, no_loss_C, no_loss_D, no_loss_E = 0, 0, 0, 0, 0
no_A, no_B, no_C, no_D, no_E = 0, 0, 0, 0, 0
for i in range(0, len(games)):
    if games["opponent"][i]=="Team A":
        no_A+=1
        if games["result"][i]=="Win":
            no_win_A+=1
        elif games["result"][i]=="Draw":
            no_draw_A+=1
        elif games["result"][i]=="Loss":
            no_loss_A+=1
    elif games["opponent"][i]=="Team B":
        no_B+=1
        if games["result"][i]=="Win":
            no_win_B+=1
        elif games["result"][i]=="Draw":
            no_draw_B+=1
        elif games["result"][i]=="Loss":
            no_loss_B+=1
    elif games["opponent"][i]=="Team C":
        no_C+=1
        if games["result"][i]=="Win":
            no_win_C+=1
        elif games["result"][i]=="Draw":
            no_draw_C+=1
        elif games["result"][i]=="Loss":
            no_loss_C+=1
    elif games["opponent"][i]=="Team D":
        no_D+=1
        if games["result"][i]=="Win":
            no_win_D+=1
        elif games["result"][i]=="Draw":
            no_draw_D+=1
        elif games["result"][i]=="Loss":
            no_loss_D+=1
    elif games["opponent"][i]=="Team E":
        no_E+=1
        if games["result"][i]=="Win":
            no_win_E+=1
        elif games["result"][i]=="Draw":
            no_draw_E+=1
        elif games["result"][i]=="Loss":
            no_loss_E+=1

cond_probabilities_A=[no_win_A/no_A, no_draw_A/no_A, no_loss_A/no_A]
cond_probabilities_B=[no_win_B/no_B, no_draw_B/no_B, no_loss_B/no_B]
cond_probabilities_C=[no_win_C/no_C, no_draw_C/no_C, no_loss_C/no_C]
cond_probabilities_D=[no_win_D/no_D, no_draw_D/no_D, no_loss_D/no_D]
cond_probabilities_E=[no_win_E/no_E, no_draw_E/no_E, no_loss_E/no_E]

print("\nProbabilitatile result|A: ", cond_probabilities_A)
print("Probabilitatile result|B: ", cond_probabilities_B)
print("Probabilitatile result|C: ", cond_probabilities_C)
print("Probabilitatile result|D: ", cond_probabilities_D)
print("Probabilitatile result|E: ", cond_probabilities_E)

print("\nEntropy of result|A: ", H(cond_probabilities_A))
print("Entropy of result|B: ", H(cond_probabilities_B))
print("Entropy of result|C: ", H(cond_probabilities_C))
print("Entropy of result|D: ", H(cond_probabilities_D))
print("Entropy of result|E: ", H(cond_probabilities_E))

H_result_opponent=(no_A/len(games)*H(cond_probabilities_A)+no_B/len(games)*H(cond_probabilities_B)+
no_C/len(games)*H(cond_probabilities_C)+no_D/len(games)*H(cond_probabilities_D)+no_E/len(games)*H(cond_probabilities_E))
print("----Entropy of result|opponent: ", H_result_opponent)

print("\n----IG of result|stadium: ", (H_result-H_result_stadium))
print("----IG of result|opponent:", (H_result-H_result_opponent))
print(games)


# Exercise 4 -- Malina Pinzariu

# Consider the random variable  𝐶  "a person has a cold" and the random variable  𝑇  "outside temperature". The joint distribution of the two variables is given below.
# 1.Plot the pmf of  𝐶  and  𝑇 .
# 2.Calculate  𝐻(𝐶) ,  𝐻(𝑇) .
# 3.Calculate  𝐻(𝐶|𝑇) ,  𝐻(𝑇|𝐶) . Does the temperature (T) reduce the uncertainty regarding someone having a cold (C)?

import pandas as pd
import math
from scipy import stats
from scipy.stats import entropy
import matplotlib.pyplot as plt

d = pd.DataFrame({'T_Sunny': [0.3, 0.05], 'T_Rainy': [0.2, 0.15], 'T_Snowy': [0.1, 0.2]},
                 index=['C_No', 'C_Yes'])

def H(p):
    log_zero = lambda x: 0 if x == 0 else math.log2(x)
    return -sum(i*log_zero(i) for i in p)

k= len(d.iloc[0])
i=0
# p = []
p = [0]*2
while i < 2:
    s=0
    for j in range(3):
        s += d.iloc[i][j]
    p[i] = s
    i += 1

t = [0]*3
i = 0
while i < 3:
    s = 0
    for j in range(2):
        s += d.iloc[j][i]
    t[i] = s
    i += 1

outcomes = {'C_No': 0, 'C_Yes': 1}
outc = sorted(set(outcomes.values()))
pmf = [p[x] for x in outc]
fig, ax = plt.subplots(1, 1)
ax.bar(outc, pmf)
plt.ylabel("pmf")
plt.xlabel("Cold")
plt.title("Probability mass function")
plt.show()

outcomes = {'t_Sunny': 0, 'T_Rainy': 1, 'T_Snowy': 2}
outc = sorted(set(outcomes.values()))
pmf = [t[x] for x in outc]
fig, ax = plt.subplots(1, 1)
ax.bar(outc, pmf)
plt.ylabel("pmf")
plt.xlabel("Time")
plt.title("Probability mass function")
plt.show()

print("\nH(C) = ", H(p))
# print(entropy(p, base=2))
print("H(T) = ", H(t))

'''
 H(C|T) H(T|C)
'''

def h(p):
    log_zero = lambda x: 0 if x == 0 else math.log2(x)
    return -p*log_zero(p)

V = [[0 for i in range(3)]
               for j in range(2)]
for i in range(3):
    for j in range(2):
        V[j][i] = d.iloc[j][i]
c1 = 0
c2 = 0
for b in range(2):
    for v in range(3):
        prob = V[b][v]/p[b]
        c1 = c1 + p[b] * h(prob)

for b in range(3):
    for v in range(2):
        prob2 = V[v][b] / t[v]
        c2 = c2 + t[v] * h(prob2)

print("H(T|C) = ", c1)
print("H(C|T) = ", c2)


# Exercise 5 -- Tcaciuc Ionut

# Consider the Boolean expression A∨(B∧C). The corresponding truth table can be generated with:
# 1.Fit a decision tree classifier on the truth table above and visualise the resulting tree. Make sure to use the entropy as a metric.
# 2.Is the tree above optimal? Can you find a decision tree with fewer levels or nodes that correctly represents this function?

import matplotlib.pyplot as plt
from itertools import product
from sklearn import tree

X = [list(c) for c in product([0,1], repeat=3)]
y = [A or (B and C) for A, B, C in X]

#a
print(X)
print(y)

dt=tree.DecisionTreeClassifier(criterion='entropy', max_depth=3).fit(X, y)
fig, ax = plt.subplots(figsize=(7, 8))
f=tree.plot_tree(dt, ax=ax, fontsize=10, feature_names=["A", "B", "C"])
plt.show()

#b
X.remove([0,1,1])
y.remove(1)
dt=tree.DecisionTreeClassifier(criterion="entropy").fit(X, y)
fig, ax = plt.subplots(figsize=(7, 8))
f=tree.plot_tree(dt, ax=ax, fontsize=10, feature_names=["A", "B", "C"])
plt.show()