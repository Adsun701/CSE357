import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import csv
import time

from sklearn.linear_model import LinearRegression


# helper functions

# standardizes an array.
def standardize(arr):
  std = sampleStd(arr)
  mu = mean(arr)
  return [(n - mu) / std for n in arr]

# returns the mean of an array.
def mean(arr):
  return sum(arr) / len(arr)

# returns the sample variance of an array.
def sampleVar(arr):
  n = len(arr)
  s = 0
  m = mean(arr)
  for i in range(n):
    s += (arr[i] - m) ** 2
  return (1 / (n - 1)) * s

# returns the sample standard deviation of an array.
def sampleStd(arr):
  return sampleVar(arr) ** 0.5

# returns the covariance of two arrays
# that are equal in length.
def covariance(arr1, arr2):
  sum = 0
  n = len(arr1)
  m1 = mean(arr1)
  m2 = mean(arr2)
  for i in range(n):
    sum += (arr1[i] - m1) * (arr2[i] - m2)
  return sum / (n - 1)

# returns the slope of a regression line given two arrays
# and a pearson correlation coefficient.
def slope(x_data, y_data, r):
  return r * (sampleStd(y_data) / sampleStd(x_data))

# returns the y-intercept of a regression line given two arrays
# and a pearson correlation coefficient.
def yIntercept(x_data, y_data, r):
  return mean(y_data) - slope(x_data, y_data, r) * mean(x_data)

def rFromSlope(slope, x_data, y_data):
  return slope * sampleStd(x_data) / sampleStd(y_data)

# residual sum of squares
def rss(arr, arrhat):
  s = 0.0
  for i in range(len(arr)):
    s += (arr[i] - arrhat[i]) ** 2
  return s

def rs(arr, arrhat):
  s = 0.0
  for i in range(len(arr)):
    s += arrhat[i] - arr[i]
  return s

def x_times_rs(xarr, yarr, yarrhat):
  s = 0.0
  for i in range(len(xarr)):
    s += xarr[i] * (yarrhat[i] - yarr[i])
  return s

# returns the degrees of freedom for a two-sample t-test
def degreesOfFreedom(v1, v2, n1, n2):
  return (((v1 / n1 + v2 / n2) ** 2) /
    ((v1 / n1)**2 / (n1 - 1) + (v2 / n2)**2 / (n2 - 1)))



# SETUP

with open("main.csv", 'r') as csv_file:
  # create a csv reader.
  csv_reader = csv.reader(csv_file)

  # for header row, check index of cases column, and death column.
  header_row = next(csv_reader)

  # read rest of rows.
  x_datas_orig = np.array(list(csv_reader))

x_datas = np.transpose(x_datas_orig)

# get non-numeric datas
geoids = x_datas[0]
state_names = x_datas[1]
city_names = x_datas[2]

# get cases_data and deaths_data
cases_data = x_datas[3].astype(float)
deaths_data = x_datas[4].astype(float)

x_datas = x_datas[5:].astype(float)

x_datas_mod = [standardize(x_data) for x_data in x_datas]

cases_data_mod = standardize(cases_data)
deaths_data_mod = standardize(deaths_data)


#####################################

## Method 1: Hypothesis Testing
def hypothesisTest(x_datas, y_data, rss, coefs, alpha):
  n = len(y_data)
  m = len(x_datas)
  df = n - (m + 1)
  v = rss / df

  # get p-values from standardized coefficients.
  for i in range(m):
    x_data = x_datas[i]
    mu = mean(x_data)
    s = sum([(x - mu) ** 2 for x in x_data])
    t = coefs[i] / (v / s) ** 0.5
    if (t < 0):
      p = ss.t.cdf(t, df)
    else:
      p = 1 - ss.t.cdf(t, df)
    print("t for b" + str(i + 1) + " is " + str(t))
    print("uncorrected p-value for b" + str(i + 1) + " is "
      + str(p))
    print("corrected p-value for b" + str(i + 1) + " is "
      + str(bonferroniCorrect(p, m)) + "\n")
  
  

#####################################

## Method 2: Bonferroni correction
def bonferroniCorrect(p, n):
  return p * n


#####################################

## Method 3: Multiple Linear Regression
# gradient descent algorithm for multiple linear regression,
# x_datas is a 2D array, y_data is a 1D array,
# and l2 is a boolean setting whether or not
# to use l2 regularization.
def grad_descent_multi(x_datas, y_data, l2=False, penalty=0.1, max_iter=1000):
  num_of_predictors = len(x_datas)
  num_of_points = len(y_data)
  prev_b0 = -1
  prev_b1s = [-1] * num_of_predictors
  b0 = 0
  b1s = [0] * num_of_predictors
  alpha = 0.0001
  timesDivided = 0
  timesOverflowed = 0
  prev_rss = np.inf
  current_rss = np.inf
  done = False
  i0 = 0
  overFlowStep = -1
  just_overflowed = False
  yhat_data = [0] * num_of_points
  while(done == False):
    for i in range(num_of_points):
      s = b0
      for j in range(num_of_predictors):
        s += b1s[j] * x_datas[j][i]
      yhat_data[i] = s
    prev_rss = current_rss
    if (l2==False): l2_term = 0
    else: l2_term = penalty * (b0 * b0 + sum([b1 * b1 for b1 in b1s]))
    current_rss = 0
    for i in range(num_of_points):
      current_rss += (y_data[i] - yhat_data[i]) ** 2
    current_rss += l2_term
    if (current_rss > prev_rss):
      just_overflowed = True
      current_rss = prev_rss
      timesOverflowed += 1
      overFlowStep = i0
      alpha /= 10
      if (timesOverflowed < 3):
        timesDivided += 1
        timesOverflowed = 0
      b0 = prev_b0
      b1s = prev_b1s.copy()
      continue
    if (just_overflowed == False and prev_rss != np.inf and abs(current_rss - prev_rss) < 0.0001):
      done = True
      break
    elif (i0 >= max_iter):
      done = True
      break
    prev_b0 = b0
    prev_b1s = b1s.copy()
    b0 = b0 - alpha * rs(y_data, yhat_data)
    for i in range(num_of_predictors):
      b1s[i] = b1s[i] - alpha * x_times_rs(x_datas[i], y_data, yhat_data)
    if (overFlowStep > 0 and i0 - overFlowStep >= 10 and timesDivided > 0):
      alpha *= 10
      timesDivided -= 1
      if (timesDivided == 0): overFlowStep = -1
    elif (i0 % 50 == 0):
      alpha *= 10
    just_overflowed = False
    i0 += 1
  return (b1s, b0, current_rss)


#####################################

## Method 4: Correlation

# returns the correlation between two arrays
# assumed to be equal in length.
def correlation(arr1, arr2):
  return covariance(arr1, arr2) / (sampleStd(arr1) * sampleStd(arr2))

#####################################

## Method 5: Bootstrap Confidence Interval
def bootstrap_ci(x_data, y_data, iters=1000):
  all_coefs = []
  for i in range(iters):
    rx = np.random.choice(x_data, size=len(x_data), replace=True)
    ry = np.random.choice(x_data, size=len(y_data), replace=True)
    coef = correlation(rx, ry)
    all_coefs.append(coef)

  sorted_coefs = sorted(all_coefs)

  l = sorted_coefs[int(0.025*iters)]
  u = sorted_coefs[-int(0.025*iters)]
  return (l, u)


# main() function
def main():
  print("Method 3: Multiple Linear Regression with Gradient Descent")
  print("Linear regression with number of cases of COVID-19 as dependent variable")
  (coefs_cases, intercept, rss) = grad_descent_multi(x_datas_mod, cases_data_mod, l2=True)
  b1s = [0] * len(x_datas)
  s = 0
  for i in range(len(b1s)):
    b1s[i] = slope(x_datas[i], cases_data, coefs_cases[i])
    s += b1s[i] * mean(x_datas[i])
  b0 = mean(cases_data) - s
  for i in range(len(b1s)):
    print("b" + str(i + 1) + " (" + str(header_row[i + 5]) + ") is", b1s[i])
  print("Y-intercept is", b0)
  print("\n")

  print("Linear regression with number of deaths from COVID-19 as dependent variable")
  (coefs_deaths, intercept, rss) = grad_descent_multi(x_datas_mod, deaths_data_mod, l2=True)
  b1s = [0] * len(x_datas)
  s = 0
  for i in range(len(b1s)):
    b1s[i] = slope(x_datas[i], deaths_data, coefs_deaths[i])
    s += b1s[i] * mean(x_datas[i])
  b0 = mean(deaths_data) - s
  for i in range(len(b1s)):
    print("b" + str(i + 1) + " (" + str(header_row[i + 5]) + ") is", b1s[i])
  print("Y-intercept is", b0)
  print("\n")

  print("Method 1: Hypothesis Testing using the T-distribution")
  print("Cases:")
  hypothesisTest(x_datas_mod, cases_data_mod, rss, coefs_cases, 0.05)
  print("\n")

  print("Deaths:")
  hypothesisTest(x_datas_mod, deaths_data_mod, rss, coefs_deaths, 0.05)
  print("\n")

  # find correlation coefficients using covariance / (sx * sy) formula
  print("Method 4: Correlation")
  print("Finding correlation coefficients using covariance / (sx * sy) formula for cases of COVID-19")
  for i in range(len(x_datas)):
    print("correlation coefficient " + str(i + 1) +
      " (" + str(header_row[i + 5]) + ") is", correlation(x_datas[i], cases_data))
  print("\n")
  print("Finding correlation coefficients using covariance / (sx * sy) formula for deaths from COVID-19")
  for i in range(len(x_datas)):
    print("correlation coefficient " + str(i + 1) +
      " (" + str(header_row[i + 5]) + ") is", correlation(x_datas[i], deaths_data))
  print("\n")


  # get bootstrap confidence intervals of each coefficient
  print("Method 5: Bootstrap Confidence Intervals")
  print("Finding bootstrap confidence intervals for correlation coefficients with predictors and cases of COVID-19")
  for i in range(len(x_datas)):
    l, u = bootstrap_ci(x_datas[i], cases_data)
    print("confidence interval for correlation coefficient " + str(i + 1) + " ("
      + str(header_row[i + 5]) + ") is [", l, ", ", u, "]")
  print("\n")
  print("Finding bootstrap confidence intervals for correlation coefficients with predictors and deaths from COVID-19")
  for i in range(len(x_datas)):
    l, u = bootstrap_ci(x_datas[i], deaths_data)
    print("confidence interval for correlation coefficient " + str(i + 1) + " ("
      + str(header_row[i + 5]) + ") is [", l, ", ", u, "]")
  print("\n")


if __name__=="__main__":
  main()
