#Exercise 1 -- Paul Boca
#(Random variables, implementation)

#Give an example of a random variable and plot an illustrative pmf using matplotlib and scipy.stats functions for the following discrete distributions:

import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.stats import geom

# Binomial distribution
# Random variable X = "Number of sixes(6) we got from 20 tosses of a die."

n = 20  # number of tosses
k_X = range(n)  # values of X
p_X = 1/6  # probability of success

pmf_X = binom.pmf(k_X, n, p_X)

fig, ax = plt.subplots(1, 1)
ax.bar(k_X, pmf_X)
plt.xlabel("k")
plt.ylabel("pmf")
plt.title("Binomial Distribution")
plt.show()

# Geometric distribution
# Random variable Y = "Number of rolls until a 6 is shown on a die."

k_Y = range(50)  # values of Y
p_Y = 1/6  # probability of success

pmf_Y = geom.pmf(k_Y, p_Y)

fig1, ax = plt.subplots(1, 1)
ax.bar(k_Y, pmf_Y)
plt.xlabel("k")
plt.ylabel("pmf")
plt.title("Geometric Distribution")
plt.show()


#Exercise 2 -- Butnariu Cristi
#(Random variables, implementation)

#Give an example of a random variable and plot an illustrative pdf using matplotlib and scipy.stats functions for the following discrete distributions:
#gamma;
#Pareto.

#Gamma

import numpy as np
from scipy.special import gamma
import pylab

ax = pylab.linspace(-5, 5, 1000)
pylab.plot(ax, gamma(ax), ls='-', c='k', label='$\Gamma(x)$')

ax2 = pylab.linspace(1,6,6)
xm1fac = np.array([1, 1, 2, 6, 24, 120])
pylab.plot(ax2, xm1fac, marker='*', markersize=12, markeredgecolor='r',
           markerfacecolor='r', ls='',c='r', label='$(x-1)!$')

pylab.ylim(-50,50)
pylab.xlim(-5, 5)
pylab.xlabel('$x$')
pylab.legend()
pylab.show()

#Pareto

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pareto
x_m = 1 #scale
alpha = [1, 2] #list of values of shape parameters
samples = np.linspace(start=0, stop=5, num=1000)#esantionul
for a in alpha:
    output = np.array([pareto.pdf(x=samples, b=a, loc=0, scale=x_m)])
    plt.plot(samples, output.T, label='alpha {0}' .format(a))
plt.xlabel('samples', fontsize=15)
plt.ylabel('PDF', fontsize=15)
plt.title('Probability Density function', fontsize=15)
plt.grid(b=True, color='grey', alpha=0.3, linestyle='-.', linewidth=2)
plt.rcParams["figure.figsize"] = [5, 5]
plt.legend(loc='best')
plt.show()


#Exercise 3 -- Cuberschi Lucian
#(Random variables, implementation)

#Suppose you measure the temperature 10 consecutive days with a thermometer that has a small random error.

# 1.What is the mean temperature, knowing that the mean error is +1°C and the measurements are those in the variable Y below.
# 2.You have a second thermometer with a Fahrenheit scale ( 𝑇(°𝐹)=𝑇(°𝐶)×1.8+32 ).
# The variance of the measurements of this thermometer (in Fahrenheit) is 8. Is this higher or lower than the variance shown by the first one?

import math
from scipy.stats import poisson

C = [21, 20, 22, 23, 20, 19, 19, 18, 19, 20]
mean = sum(C)/len(C)-1
print('mean 1 is: ', poisson.mean(mean))

C1 = [ x*18/10 +32 - (1*18/10 +32) for x in C ]
meanF = sum(C1)/len(C1)
sd = 0.0
for i in C1:
    sd = sd + (i - meanF)*(i - meanF)#sum of x - mean(x)
sd = math.sqrt(sd/9)#sd formula
varC = sd*sd
print('varC is: ',varC)
varF = 8
if varF>varC:
    print('VarF higher then VarC')
else:
    print('VarF lower then VarC')


#Exercise 4 -- Cosmin Cruceanu
#(Random variable, implementation)

#Let  𝑆  be the outcome of a random variable describing the sum of two dice thrown independently.

#Print the probability distribution of  𝑆  graphically.
#Determine  𝐸[𝑆]  and  𝑉𝑎𝑟(𝑆) .

from itertools import product
from typing import Set, Any
import statistics

import matplotlib.pyplot as plt


def probability(A: Set[Any], omega: Set[Any]):
    """ Probability for a uniform distribution
    in a finite space"""
    return len(A) / len(omega)


def expectation(values):
    return sum(_i * values[_i] for _i in values)


def mean(values):
    return sum(values) / len(values)


def variance(values):
    _mean = mean(values)
    return sum(S[x] * ((x - _mean) ** 2) for x in values)


_omega = set(product(range(1, 7), range(1, 7)))
S = {}
for i in range(2, 13):
    temp = set((a, b) for a, b in _omega if a + b == i)
    S[i] = probability(temp, _omega)
k = sorted(set(S.values()))
pmf_S = [x for pair in S for x in k if S[pair] == x]
fig, ax = plt.subplots(1, 1)
ax.bar(S.keys(), pmf_S)
plt.ylabel("probability")
plt.xlabel("sum of dice")
plt.show()
print(statistics.mean(S.keys()))
print(expectation(S))
print(variance(S))


#Exercise 5
#(Random variable, conceptual)

#The probability distribution of a discrete random variable 𝑋 is given by 𝑃(𝑋=−1)=1/5,𝑃(𝑋=0)=2/5,𝑃(𝑋=1)=2/5.

# 1.Compute 𝐸[𝑋].
# 2.Give the probability distribution of 𝑌=𝑋2 and compute 𝐸[𝑌] using the distribution of 𝑌.
# 3.Determine 𝐸[𝑋2] using the change-of-variable formula. Check your answer against the answer in 2.
# 4.Determine 𝑉𝑎𝑟(𝑋).


# L-am facut corect in caiet


#Exercise 6 -- Iftime Stefan
#(binomial distribution, applied)

# A sailor is trying to walk on a slippery deck, but because of the movements of the ship, he can make exactly one step every second,
# either forward (with probability  𝑝=0.5 ) or backward (with probability  1−𝑝=0.5 ). Using the scipy.stats.binom package,
# determine the probability that the sailor is in position +8 after 16 seconds.

from scipy.stats import binom

k = 12
n = 16
p = 0.5
Pmf = binom.pmf(k, n , p)
print(Pmf)


#Exercise 7 -- Andreea Munteanu
#(geometric distribution, applied)

# In order to finish a board game, a player must get an exact 3 on a regular die. Using the scipy.stats.geom package,
# determine how many tries will it take to win the game (on average)? What are the best and worst cases?

import numpy as np
import scipy.stats as stats
from scipy.stats import geom
import matplotlib.pyplot as plt

# S     |   1 |   2 |   3 |   4 |   5 |   6
# P(S)  | 1/6 | 1/6 | 1/6 | 1/6 | 1/6 | 1/6 |


die = stats.randint(1, 7) # die can take any value in [1,6]
p = die.pmf(3)            # probability for die to be 3 at 1 roll: 0.16666666...
# print("probability = ", p)

"""
r = range(1, 7)
# distribution object for which the probability of happenstance is p = 0.1(6)%:
die_is_3 = stats.geom(p)
probabilities = [die_is_3.pmf(i) for i in r]

die_sample = np.random.randint(1, 7, size=100)

plt.style.use('bmh')
fig, ax = plt.subplots()
ax.set_title('Geometric distribution')
plt.hist(die_sample, bins=50)
plt.bar(r, probabilities)
"""
mean, var = geom.stats(p, moments='mv')

print("mean(geom) = ", mean, ", mean(formula) = ", 1/p)
print("variance(geom) = ", var, ", by formula = ",  (1-p)/(p*p))
plt.show()


#Exercise 8 -- Delia Stoica
#(Normal distribution, applied)

# The grades from an exam roughly follow a Gaussian distribution of mean 5 and variance 2.
# Using the scipy.stats.norm package, determine what percentage of students will pass the exam, if the minimum score is 3.

from statistics import NormalDist
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

mu = 5
sigma = 2
x = np.linspace(-20, 20)
pdf_X = norm.pdf(x, mu, sigma)

mean = norm.mean(mu, sigma)

print('mean is', mean)

fig, ax = plt.subplots(1, 1)
ax.plot(x, pdf_X)
ax.vlines(mean, 0, max(pdf_X), colors='pink')
plt.ylabel("pdf")
plt.xlabel("x")
plt.title("Probability density function")
plt.show()

#print(NormalDist(mu=5, sigma=2).pdf(3))
#print(NormalDist(mu=5, sigma=2).cdf(3))

percentage = 1-NormalDist(mu=5, sigma=2).cdf(3)
p2 = 1 - norm.cdf(3, loc=5, scale=2)

print("The percentage of the students that will pass the exam is ", percentage)
print("The percentage of the students that will pass the exam is ", p2)