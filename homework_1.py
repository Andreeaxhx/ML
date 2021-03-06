#Exercise 3
#(Events, quick, implementation)

#Illustrate DeMorgan's laws using the plot_venn() function and standard Python set operations:
#¬(𝐴∪𝐵) =¬𝐴∩¬𝐵
#¬(𝐴∩𝐵) =¬𝐴∪¬𝐵

from tools.venn import A, B, omega, plot_venn
# First law
plot_venn(omega - (A.union(B)))
plot_venn((omega - A).intersection(omega - B))
# Second law
plot_venn(omega - (A.intersection(B)))
plot_venn((omega - A).union(omega - B))

#Exercise 4 -- Eu
#(Product of sample spaces, quick, implementation)

#Two dice are thrown simultaneously. Calculate the probability that the sum is 11.

from itertools import product

A = set(product([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]))
B = []
print("Cazuri posibile: ", A)

# for ul se putea scrie intr-o linie folosind liste
# [i for i in dice if i[0]+i[1]==11]

for i in A:
    if i[0] + i[1] == 11:
        B.append(i)
B = set(B)
print("Cazuri favorabile: ", B)
print("Probabilitatea ca suma celor doua zaruri sa fie 11: ", len(B) / len(A))

#Exercise 4 -- Doru Alexandru

from itertools import product
# Generam toate combinatiile de zaruri posibile
dice = list(product(range(1, 7), repeat=2))
# Lista in care se stocheaza combinatiile care dau suma = 11
combs = list()
for i in dice:
    if i[0] + i[1] == 11:
        combs.append(i)
# combs = [i for i in dice if i[0] + i[1] == 11]

print('Toate combinatiile posibile: ', dice)
print('Combinatiile de zaruri care au suma 11:', combs)
print('Probabilitatea ca suma zarurilor sa fie egala cu 11:', len(combs) / len(dice))

#Exercise 7 - Monty Hall problem
#(Bayes' theorem, normal, implementation, analysis)

#Suppose you are in a game show and you're given the choice of three doors; behind one is a car, behind the others, goats.
#You pick door no. 1, but don't open it. The game host (who know what is behind each door) then opens a door which always
# has a goat (in this case opens door no. 2) and asks you if you still want to open door no.1 or you want to switch to no.3.

#What are the probabilities of finding the car in the two cases?

#1.Create a Python simulation for 1000 games to estimate the answer.
#2.Find the answer using the tool.stats.probability_weighted function (see this approach for constructing the sample space).
#3.Find the answer mathematically by applying Bayes' theorem and the law of total probability.

# varianta 2
@dataclass(frozen=True)
class Outcome(WeightedOutcome):
    type: str
    case: str


omega = {Outcome(type='switch', case='ABC', weight=1 / 9), Outcome(type='switch', case='ACB', weight=1 / 9),
         Outcome(type='switch', case='BAC', weight=1 / 9), Outcome(type='switch', case='BCA', weight=1 / 9),
         Outcome(type='switch', case='CAB', weight=1 / 9), Outcome(type='switch', case='CBA', weight=1 / 9),
         Outcome(type='maintain', case='AAB', weight=1 / 18), Outcome(type='maintain', case='AAC', weight=1 / 18),
         Outcome(type='maintain', case='BBA', weight=1 / 18), Outcome(type='maintain', case='BBC', weight=1 / 18),
         Outcome(type='maintain', case='CCA', weight=1 / 18), Outcome(type='maintain', case='CCB', weight=1 / 18)}

A = set(o for o in omega if o.type is 'switch')
B = set(o for o in omega if o.type is 'maintain')
print(f"The probability to win if the player switch the door is {probability_weighted(A, omega)}")
print(f"The probability to win if the player don't switch the door is {probability_weighted(B, omega)}")

#Exercise 12
#(Independent events, normal, implementation)

#Consider the event space corresponding to two tosses of a fair coin, and the events A "heads on toss 1",
#B "heads on toss 2" and C "the two tosses are equal". Using the tools.stats.probability function, find if:

#events A and B are independent;
#events A and C are independent.

from tools.stats import probability
# Code here
from itertools import product
from tools.stats import probability

omega_1 = set(['H', 'T'])
omega_2 = set(['H', 'T'])
omega = set(product(omega_1, omega_2))
A = set(a for a in omega if 'H' == a[0])
B = set(a for a in omega if 'H' == a[1])
C = set(a for a in omega if a[0] == a[1])
print("omega: ", omega)
print("A (heads on toss 1): ", A)
print("B (heads on toss 2): ", B)
print("C (the two tosses are equa: )", C)

p1 = probability(A, omega) * probability(B, omega)
p2 = probability(A.intersection(B), omega)

if p1 == p2:
    print("events A and B are independent")
else:
    print("events A and B are NOT independent")

p1 = probability(A, omega) * probability(C, omega)
p2 = probability(A.intersection(C), omega)

if p1 == p2:
    print("events A and C are independent")
else:
    print("events A and C are NOT independent")

#varianta prof
from itertools import product
from tools.stats import probability

omega_1 = set(['H', 'T'])
omega_2 = set(['H', 'T'])
omega = set(product(omega_1, omega_2))

# Create event 'heads on toss 1'
A = set(a for a in omega if a[0] is 'H')
print('A:', A)
# Create event 'heads on toss 2'
B = set(a for a in omega if a[1] is 'H')
print('B:', B)
# Create event 'the two tosses are equal'
C = set(a for a in omega if a[0] is a[1])
print('C:', C)

print('P(A)=', probability(A, omega))
print('P(A|B)=', probability(A & B, B))  # Yes, they are because P(A)=P(A|B)
print('P(A|C)=', probability(A & C, C))  # Yes, they are because P(A)=P(A|C)