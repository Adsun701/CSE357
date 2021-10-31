import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import sys

#sys.stdout = open('a1_sun_112532970_OUTPUT.txt', 'w')


# PART 1
# this is the array to store the counts of d1. Index n - 1 stores the value n.
a1 = np.array([0, 0, 0, 0, 0, 0])
# sum array. Index n - 1 stores the value n.
sumA = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# double array. the first index is the value of d1 - 1. The first index
# will point to the nested array that stores a 12-long array of probabilities
# of d1 + d2 and d1.
doubleA = np.zeros((6, 12))

n = 1000000
for i in range(n):
  d1 = np.random.randint(1, 7)
  d2 = np.random.randint(1, 7)
  sum = d1 + d2
  a1[d1 - 1] += 1
  sumA[sum - 1] += 1
  doubleA[d1 - 1][sum - 1] += 1

for i in range(len(a1)):
  doubleA[i] = doubleA[i] / n

a1 = a1 / n
sumA = sumA / n

p1 = a1[5]
p2 = sumA[4]
prod = p1 * p2
actualProb = doubleA[5][4]
equalChar = '0'
if abs(prod - actualProb) <= 0.01:
  equalChar = '≈'
else:
  equalChar = '≠'
print("P(D1 = 6) = %f" % p1)
print("P(D1 + D2 = 5) = %f" % p2)
print("P(D1 = 6) * P(D1 + D2 = 5) = %f" % prod)
print("P(D1 = 6, D1 + D2 = 5) = %f" % actualProb)
print("P(D1 = 6) * P(D1 + D2 = 5)", equalChar, "P(D1 = 6, D1 + D2 = 5)\n")

p1 = a1[4]
p2 = sumA[5]
prod = p1 * p2
actualProb = doubleA[4][5]
if abs(prod - actualProb) <= 0.01:
  equalChar = '≈'
else:
  equalChar = '≠'
print("P(D1 = 5) = %f" % p1)
print("P(D1 + D2 = 6) = %f" % p2)
print("P(D1 = 5) * P(D1 + D2 = 6) = %f" % prod)
print("P(D1 = 5, D1 + D2 = 6) = %f" % actualProb)
print("P(D1 = 5) * P(D1 + D2 = 6)", equalChar, "P(D1 = 5, D1 + D2 = 6)\n")

p1 = a1[3]
p2 = sumA[6]
prod = p1 * p2
actualProb = doubleA[3][6]
if abs(prod - actualProb) <= 0.01:
  equalChar = '≈'
else:
  equalChar = '≠'
print("P(D1 = 4) = %f" % p1)
print("P(D1 + D2 = 7) = %f" % p2)
print("P(D1 = 4) * P(D1 + D2 = 7) = %f" % prod)
print("P(D1 = 4, D1 + D2 = 7) = %f" % actualProb)
print("P(D1 = 4) * P(D1 + D2 = 7)", equalChar, "P(D1 = 4, D1 + D2 = 7)\n")

# PART 2
mean = 42
s = 8.5
a = ss.norm(mean, s)
x_range = np.linspace(10, 74, 1000)
y = [a.pdf(x_value) for x_value in x_range]
y_cdf = [a.cdf(x_value) for x_value in x_range]

# plot our distribution's pdf
plt.plot(x_range, y)
plt.show()
plt.savefig("normal_pdf_plot.png")

plt.plot(x_range, y_cdf)
plt.show()
plt.savefig("normal_cdf_plot.png")

# Find out probability that phone will die in 330
# minutes given that phone was charged 34 hours ago.
# AKA Find the probably that the phone's battery life
# is less than 34 + 5.5 = 39.5 hours given that 34 hours
# have passed.
probabilityPhoneWillDie = (a.cdf(39.5) - a.cdf(34)) / (1 - a.cdf(34))
print("The probability that the phone will die before 39.5 hours but after 34 hours:", probabilityPhoneWillDie)
densityValue = a.pdf(39.5)
print("Density value is", densityValue)

# Find out how many hours you should insist she texts
# by, assuming 95% confidence that your phone won't be
# dead before she texts.
hours = a.ppf(a.cdf(34) + 0.05)
print("total # of hours you insist she should text by from hour 0 (assuming 95% confidence that your phone won't be dead by then):", hours)
print("She should text you by", (hours - 34), "hours.")

# Find probability that phone dies while
# you're hanging out with friend for 2.5 hours.
pPhoneDiesWithFriend = (a.cdf(hours + 3) - a.cdf(hours + 0.5))/(1 - a.cdf(hours + 0.5))
print("Probability phone dies while you're hanging out with friend:", pPhoneDiesWithFriend)
x2_range = np.linspace(hours + 0.5, hours + 3, 1000)
y2_cdf = [a.cdf(x2_value) for x2_value in x2_range]
plt.plot(x2_range, y2_cdf)
plt.show()
plt.savefig("friend_hangout_cdf_plot.png")

# PART 3

def randKumaraswamy(size = 100, a = 2, b = 2):
  #returns a list of size of random numbers drawn from a Kumaraswamy(a, b) distribution
  uniform_sample = np.random.random_sample(size)
  for i in range(size):
    x = uniform_sample[i]
    uniform_sample[i] = (a * b * x ** (a - 1)) * (1 - x ** a) ** (b - 1)
  return uniform_sample

# create a kernel density estimation with randKumaraswamy()
arr = randKumaraswamy()
estimated_pdf = ss.gaussian_kde(arr)
x_range3 = np.linspace(-1, 2.5, 1000)
y3 = [estimated_pdf(x_value) for x_value in x_range3]
plt.plot(x_range3, y3)
plt.plot(arr, [0]*len(arr), "X", alpha=.1)
plt.show()
