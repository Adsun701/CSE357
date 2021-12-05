import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import csv
import time

from sklearn.linear_model import LinearRegression

with open("main.csv", 'r') as csv_file:
  # create a csv reader.
  csv_reader = csv.reader(csv_file)

  row_null = []
  row_test = []

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

#####################################

## Method 1: Hypothesis Testing
def hypothesisTest(main_data, alpha):
  #main_data: the csv file containing our data
  #alpha: significance level
  with open(main_data, 'r') as csv_file:
    # create a csv reader.
    csv_reader = csv.reader(csv_file)

    row_null = []
    row_test = []

    cases_col = -1
    deaths_col = -1

    # for header row, check index of cases column, and death column.
    header_row = next(csv_reader)
    for i in range(len(header_row)):
      if header_row[i] == "cases": cases_col = i
      elif header_row[i] == "deaths": deaths_col = i

    # if any index is -1, a required column was not found. exit if so.
    if (cases_col < 0 or deaths_col < 0):
      print("Error: inputed csv file does not contain \"cases\" or \"deaths\" column.")
      sys.exit(1)

    # read rest of rows.
    x_datas_orig = np.array(list(csv_reader))

    # get transpose.
    x_datas = np.transpose(x_datas_orig)

    # get non-numeric datas
    geoids = x_datas[0]
    state_names = x_datas[1]
    city_names = x_datas[2]

    # get cases_data and deaths_data
    cases_data = x_datas[3].astype(float)
    deaths_data = x_datas[4].astype(float)

    # pop the first 5 variables from x_datas,
    # we don't need them.
    x_datas = x_datas[5:].astype(float)
    x_datas_mod = [standardize(x_data) for x_data in x_datas]

    cases_data_mod = standardize(cases_data)

    (coefs, intercept, rss) = grad_descent_multi(x_datas_mod, cases_data_mod, l2=True)

    b1s = [0] * len(x_datas)
    s = 0
    for i in range(len(b1s)):
      b1s[i] = slope(x_datas[i], cases_data, coefs[i])
      s += b1s[i] * mean(x_datas[i])
    b0 = mean(cases_data) - s
    for i in range(len(b1s)):
      print("b" + str(i + 1) + " is", b1s[i])
    print("Y-intercept is", b0)

    # get populations for test and null schools.
    #pop_test = float(row_test[population_col])
    #pop_null = float(row_null[population_col])

    # get numbers of those infected in each school.
    #cases_test = float(row_test[cases_col])
    #cases_null = float(row_null[cases_col])

    # get means.
    #mu_test = cases_test / pop_test
    #mu_null = cases_null / pop_null

    # get variances.
    #variance_test = variance(pop_test - cases_test, cases_test)
    #variance_null = variance(pop_null - cases_null, cases_null)

    # get degrees of freedom.
    #degrees = degreesOfFreedom(variance_test, variance_null, pop_test, pop_null)

    # calculate t-statistic.
    #tStat = (mu_test - mu_null) / (variance_test / pop_test + variance_null / pop_null) ** 0.5

    # calculate p-value using tStat and cdf.
    #p = 1 - ss.t.cdf(tStat, degrees)
    #if p < alpha: decision = "reject"
    #else: decision = "accept"

  #return p, decision

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

#####################################

## Method 5: Bootstrap Confidence Interval

# main() function
def main():
  print("This is main:")
  hypothesisTest("main.csv", 0.05)


if __name__=="__main__":
  main()
