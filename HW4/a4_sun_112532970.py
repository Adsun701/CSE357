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
def hypothesisTest(h0, ha):
  pass

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
def grad_descent_multi(x_datas, y_data, l2=False, penalty=0.1, max_iter=50000):
  num_of_predictors = len(x_datas)
  num_of_points = len(y_data)
  prev_b0 = -1
  prev_b1s = [-1] * num_of_predictors
  b0 = 0
  b1s = [0] * num_of_predictors
  alpha = 0.000001
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
    if (i0 > 5000 and just_overflowed == False and prev_rss != np.inf and abs(current_rss - prev_rss) < 0.000001):
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


if __name__=="__main__":
  main()
