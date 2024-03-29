import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import csv
import time

from sklearn.datasets import load_wine
wines = load_wine()

#print(wines.DESCR)

#print("alcohol", wines.data[:,0])

#print('(y, [x])', list(zip(wines.target[:5], wines.data[:5])))

is_class1 = (wines.target==1).astype(int)

np.seterr(all='raise')

#print('(is_class1, target)', list(zip(is_class1[:5], #wines.target[:5])))

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

# gradient descent algorithm, returns the slope
# and y-intercept respectively of the linear regression
# equation
def grad_descent(x_data, y_data):
  prev_b0 = -1
  prev_b1 = -1
  b0 = 0
  b1 = 0
  alpha = 0.000001
  timesDivided = 0
  timesOverflowed = 0
  prev_rss = np.inf
  current_rss = np.inf
  done = False
  i0 = 0
  overFlowStep = -1
  num_of_points = len(y_data)
  yhat_data = [0] * num_of_points
  while(done == False):
    for i in range(num_of_points):
      yhat_data[i] = b0 + b1 * x_data[i]
    prev_rss = current_rss
    current_rss = rss(y_data, yhat_data)
    if (current_rss > prev_rss):
      timesOverflowed += 1
      overFlowStep = i0
      alpha /= 10
      if (timesOverflowed < 3):
        timesDivided += 1
        timesOverflowed = 0
      b0 = prev_b0
      b1 = prev_b1
      continue
    if (prev_rss != np.inf and abs(current_rss - prev_rss) < 0.000001):
      done = True
      break
    prev_b0 = b0
    prev_b1 = b1
    b0 = b0 - alpha * rs(y_data, yhat_data)
    b1 = b1 - alpha * x_times_rs(x_data, y_data, yhat_data)
    if (overFlowStep > 0 and i0 - overFlowStep >= 5 and timesDivided > 0):
      alpha *= 10
      timesDivided -= 1
      if (timesDivided == 0): overFlowStep = -1
    elif (i0 % 20 == 0):
      alpha *= 10

    i0 += 1
  return (b1, b0, current_rss)

def sigmoid(x, b0, b1):
  return (np.e ** (b0 + b1 * x)) / (1 + np.e **(b0 + b1 * x))

# gradient descent algorithm for logistic regression,
# returns the coefficient and y-intercept respectively of
# the logistic regression equation
def grad_descent_log(x_data, y_data):
  prev_b0 = -1
  prev_b1 = -1
  b0 = 0
  b1 = 0
  alpha = 0.000001
  timesDivided = 0
  timesOverflowed = 0
  prev_rss = np.inf
  current_rss = np.inf
  done = False
  i0 = 0
  overFlowStep = -1
  num_of_points = len(y_data)
  yhat_data = [0] * num_of_points
  while(done == False):
    yhat_data = [sigmoid(x, b0, b1) for x in x_data]
    prev_rss = current_rss
    current_rss = rss(y_data, yhat_data)
    if (current_rss > prev_rss):
      timesOverflowed += 1
      overFlowStep = i0
      alpha /= 10
      if (timesOverflowed < 3):
        timesDivided += 1
        timesOverflowed = 0
      b0 = prev_b0
      b1 = prev_b1
      continue
    if (prev_rss != np.inf and abs(current_rss - prev_rss) < 0.000001):
      done = True
      break
    prev_b0 = b0
    prev_b1 = b1
    b0 = b0 - alpha * rs(y_data, yhat_data)
    b1 = b1 - alpha * x_times_rs(x_data, y_data, yhat_data)
    if (overFlowStep > 0 and i0 - overFlowStep >= 5 and timesDivided > 0):
      alpha *= 10
      timesDivided -= 1
      if (timesDivided == 0): overFlowStep = -1
    elif (i0 % 20 == 0):
      alpha *= 10

    i0 += 1
  return (b1, b0)

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

# gradient descent algorithm for multinomial
# log regression, x_datas is a 2D array, y_data
# is a 1D array, and l2 is a boolean setting
# whether or not to use l2 regularization.
def grad_descent_log_multi(x_datas, y_data, l2=False, penalty=0.1, max_iter=50000):
  num_of_predictors = len(x_datas)
  num_of_points = len(y_data)
  prev_b0 = -1
  prev_b1s = [-1] * num_of_predictors
  b0 = 0
  b1s = [0] * num_of_predictors
  alpha = 0.000000001
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
      yhat_data[i] = 1.0 / (1 + np.e ** -(s))
    prev_rss = current_rss
    if (l2==False): l2_term = 0
    else: l2_term = penalty * (b0 * b0 + sum([b1 * b1 for b1 in b1s]))
    current_rss = 0
    for i in range(num_of_points):
      current_rss += (y_data[i] - yhat_data[i]) ** 2
    current_rss += l2_term
    if (current_rss > prev_rss):
      just_overflowed = True
      timesOverflowed += 1
      overFlowStep = i0
      alpha /= 10
      if (timesOverflowed < 3):
        timesDivided += 1
        timesOverflowed = 0
      b0 = prev_b0
      b1s = prev_b1s.copy()
      continue
    if (i0 > 500 and just_overflowed == False and prev_rss != np.inf and abs(current_rss - prev_rss) < 0.00000000000001):
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

# Part 1
def part1():
  # A: Generate a scatter plot between alcohol
  # and color intensity. Alcohol is the first
  # data variable (data[:,0]) and colour
  # intensity is the 10th (data[:,9]).
  # Copy the plot to your output document.
  print("\nPART 1 A:\n")
  intensity = wines.data[:,9]
  y_data = wines.data[:,0]
  x_space = np.linspace(min(intensity) - 1, max(intensity) + 1, 50)
  plt.scatter(intensity, y_data, c='orange', label='data', alpha=0.8)
  plt.xlabel("color intensity")
  plt.ylabel("alcohol")
  plt.title("intensity effect on alcohol")
  plt.legend()
  plt.show()

  # B: Correlate alcohol with the other 12
  # attributes using the covariance formula.
  # The output should be the same as the
  # Pearson Product-Moment Correlation Coefficient
  # between alcohol (data[:,0]) and the other
  # variable (data[:,i] where i ∊ [1, 12]).
  # Print the 12 correlations.
  print("\nPart 1 B:\n")
  alc_std = sampleStd(y_data)
  for i in range(1, 13):
    x_data = wines.data[:,i]
    cov_data = covariance(x_data, y_data)
    x_std = sampleStd(x_data)
    r = cov_data / (x_std * alc_std)
    print("Correlation coefficient r" + str(i), "is", r)


  # C. Correlate alcohol with the other 12
  # attributes using simple linear regression
  # with gradient descent. Remember, to get the
  # equivalent of a "correlation" (i.e. a Pearson
  # Product-Moment Correlation Coefficient), one
  # must standardize both the outcome and the
  # predictor. Print the 12 correlations.
  print("\nPart 1 C:\n")
  for i in range(1, 13):
    x_data = wines.data[:,i]
    x_data_mod = standardize(x_data)
    y_data_mod = standardize(y_data)

    b1, b0, rss = grad_descent(x_data_mod, y_data_mod)
    r = b1
    print("Correlation coefficient r" + str(i), "is", r)

  # D. Relate all 12 attributes to alcohol at once
  # using multiple linear regression. Fit a multiple
  # linear regression model with alcohol (data[:,0])
  # as the outcome and the 12 other data variables as
  # the predictors using gradient descent. Run this two
  # ways: (1) where all the variables are standardized
  # and (2) where none of the variables are standardized.
  # Print the coefficients for all 12 variables as well
  # as the value of the intercept (ꞵ0) for both versions.
  print("\nPart 1 D:\n")
  print("With standardization")
  x_datas_mod = [standardize(wines.data[:,i]) for i in range(1, 13)]
  y_data_mod = standardize(y_data)
  b1s, b0, rss = grad_descent_multi(x_datas_mod, y_data_mod, l2=False)
  cc1 = b1s # standardized coefficients
  for i in range(len(cc1)):
    print("Correlation coefficient r" + str(i + 1) + " is", cc1[i])
  print("Y-intercept is", b0)
  x_datas_standardized = x_datas_mod

  print("\nWithout standardization")
  x_datas = [wines.data[:,i] for i in range(1, 13)]
  b1s, b0, rss = grad_descent_multi(x_datas, y_data, l2=False)
  cc2 = b1s # standardized coefficients
  for i in range(len(cc2)):
    print("Correlation coefficient r" + str(i + 1) + " is", cc2[i])
  print("Y-intercept is", b0)
  #b1s = [0] * len(x_datas)
  #s = 0
  #for i in range(len(b1s)):
  #  b1s[i] = slope(x_datas[i], y_data, cc1[i])
  #  s += b1s[i] * mean(x_datas[i])
  #b0 = mean(y_data) - s
  #for i in range(len(b1s)):
  #  print("b" + str(i + 1) + " is", b1s[i])
  #print("Y-intercept is", b0)

  # E. Test coefficients from 1.D. for significance.
  # Use the t-test of regression coefficients to find
  # the p-value for each coefficient from 1.D. For both
  # the standardized (1) and non-standardized (2)
  # versions, print the original p-values as well
  # as Bonferonni corrected p-values.
  print("\nPart 1 E:\n")
  # get N
  n = len(y_data) # should be 178
  m = len(x_datas_standardized) # should be 12
  df = n - (m + 1)
  v = rss / df
  # original p-values for standardized coefficients
  for i in range(m):
    x_data = x_datas_standardized[i]
    mu = mean(x_data)
    s = sum([(x - mu) ** 2 for x in x_data])
    t = cc1[i] / (v / s) ** 0.5
    if (t < 0):
      p = ss.t.cdf(t, df)
    else:
      p = 1 - ss.t.cdf(t, df)
    print("t for b" + str(i + 1) + " is " + str(t))
    print("uncorrected p-value for b" + str(i + 1) + " is "
      + str(p))
    print("corrected p-value for b" + str(i + 1) + " is "
      + str(p * m) + "\n")



# Part 2
def part2():
  # A. Generate a scatter plot between is_class1 and
  # colour intensity. Colour intensity is the 10th
  # (data[:,9]). Copy the plot to your output document.
  print("\nPart 2 A:\n")
  intensity = wines.data[:,9]
  x_space = np.linspace(min(intensity) - 1, max(intensity) + 1, 50)
  plt.scatter(intensity, is_class1, c='orange', label='data', alpha=0.5)
  plt.xlabel("color intensity")
  plt.ylabel("is_class1")
  plt.title("intensity effect on is_class1")
  plt.legend()
  plt.show()

  y_data = is_class1

  # B. Relate is_class1 with the other 13 attributes,
  # independently, using standardized logistic
  # regression.  Print the 12 coefficients.
  print("\nPart 2 B:\n")
  for i in range(0, 13):
    x_data = standardize(wines.data[:,i])
    b1, b0 = grad_descent_log(x_data, y_data)
    print("coefficient " + str(i + 1) + " is " + str(b1))

  # C. Relate all 13 attributes to is_class1 at once
  # using multiple logistic regression. Fit a multiple
  # logistic regression model with is_class1 as the
  # outcome and the 13 attributes as the predictors using
  # gradient descent over log loss (i.e. the inverse of
  # log likelihood). Run this two ways: (1) where all
  # the variables are standardized and (2) where none of
  # the variables are standardized.  Print the
  # coefficients for all 12 variables as well as
  # the value of the intercept (ꞵ0) for both versions.
  print("\nPart 2 C:\n")
  print("With standardization")
  x_datas = [standardize(wines.data[:,i]) for i in range(0, 13)]

  b1s, b0, _ = grad_descent_log_multi(x_datas, y_data)
  cc1 = b1s
  for i in range(len(cc1)):
    print("Correlation coefficient r" + str(i + 1) + " is", cc1[i])
  print("Y-intercept is", b0)

  print("\nWithout standardization")
  x_datas = [wines.data[:,i] for i in range(0, 13)]
  b1s, b0, _ = grad_descent_log_multi(x_datas, y_data)
  cc2 = b1s
  for i in range(len(cc2)):
    print("b" + str(i + 1) + " is", cc2[i])
  print("Y-intercept is", b0)
  #b1s = [0] * len(x_datas)
  #s = 0
  #for i in range(len(b1s)):
  #  b1s[i] = cc1[i] / ((np.sqrt(3) / np.pi) * sampleStd(x_datas[i]))
  #b0 = b0 / ((np.sqrt(3) / np.pi) * sampleStd(y_data))
  #for i in range(len(b1s)):
  #  print("b" + str(i + 1) + " is", b1s[i])
  #print("Y-intercept is", b0)

  return

# Part 3
def part3():
  # A. Setup your data for cross-validation. Use
  # sklearns train test split to make a random 80%
  # train and the remaining 20% test.
  print("\nPart 3 A:\n")
  from sklearn.model_selection import train_test_split

  Xtrain, Xtest, ytrain, ytest = \
    train_test_split(wines.data[:,1:], wines.data[:,0], test_size=0.20, random_state=42)
  print("Xs: ", Xtrain.shape, Xtest.shape, ";  ys: ", ytrain.shape, ytest.shape)

  # B. Predict alcohol content from the other 12
  # variables using a linear model.  Fit a linear
  # regression model over the training data to predict
  # alcohol (ytrain) from the other variables (Xtrain),
  # with and without L2 regularization. Try a few
  # penalties for L2 regularized version. Print both
  # the mean sequared error and the Pearson correlation
  # between your predictions and the ytest.
  print("\nPart 3 B:\n")
  XtrainMod = np.transpose(Xtrain)
  for i in range(len(XtrainMod)):
    XtrainMod[i] = standardize(XtrainMod[i])
  XtestMod = np.transpose(Xtest)
  for i in range(len(XtestMod)):
    XtestMod[i] = standardize(XtestMod[i])
  ytrainMod = standardize(ytrain)
  ytestMod = standardize(ytest)
  penalties=[0, 0.01, 0.1, 1, 10, 100]
  for penalty in penalties:
    b1s, b0, rss = grad_descent_multi(XtrainMod, ytrainMod, l2=True, penalty=penalty)
    yhats = []
    if (penalty == 0):
      print("without L2 regularization penalty:")
    else: print("for penalty "  + str(penalty) + ":\n")
    mse = 0
    yhat = 0
    for i in range(len(ytestMod)):
      yhat = b0
      for j in range(len(XtestMod)):
        yhat += XtestMod[j][i] * b1s[j]
      yhats.append(yhat)
      y = ytestMod[i]
      mse += (y - yhat) ** 2
    mse /= len(ytestMod)
    print("Mean-squared error for penalty " + str(penalty) + " is %f" % mse)
    r = covariance(yhats, ytestMod) / (sampleStd(yhats) * sampleStd(ytestMod))
    print("Pearson correlation coefficient for penalty " + str(penalty) + " is " + str(r))
    print()


  # C. Setup your data for cross-validation with is_class1.
  # Make a random 80% train and the final 20% test as follows:
  print("\nPart 3 C:\n")
  Xtrain, Xtest, ytrain, ytest = train_test_split(wines.data, is_class1, test_size=0.20, random_state=42)
  print("Xs: ", Xtrain.shape, Xtest.shape, ";  ys: ", ytrain.shape, ytest.shape)

  # D. Predict is_class1 from the 13 attributes variables
  # using logistic regression.  Fit a logistic regression model
  # to the training data to predict is_class1 (ytrain) from the
  # other variables (Xtrain), with and without L2 regularization.
  # Try a few penalties for L2 regularized version and choose the
  # best. Print the following metrics for both versions:
  # -> log loss
  # -> precision (positive predictive value)
  # -> recall (sensitivity;  true positive rate)
  # -> f1 value (2 * (precision * recall) / (precision + recall))
  # -> specificity (true negative rate)
  print("\nPart 3 D:\n")
  XtrainMod = np.transpose(Xtrain)
  for i in range(len(XtrainMod)):
    XtrainMod[i] = standardize(XtrainMod[i])
  XtestMod = np.transpose(Xtest)
  for i in range(len(XtestMod)):
    XtestMod[i] = standardize(XtestMod[i])
  penalties=[0, 0.01, 0.1, 1, 10, 100]
  nonzeroPenalties = []
  losses = []
  precisions = []
  recalls = []
  f1s = []
  specificities = []
  for penalty in penalties:
    fp = 0 # false positives.
    fn = 0 # false negatives.
    tp = 0 # true positives.
    tn = 0 # true negatives.
    b1s, b0, rss = grad_descent_log_multi(XtrainMod, ytrain, l2=True, penalty=penalty)
    yhats = []
    if (penalty == 0):
      print("without L2 regularization penalty:")
    else: print("for penalty "  + str(penalty) + ":")
    print("Log loss is", rss)
    yhat = 0
    for i in range(len(ytest)):
      y = ytest[i]
      yhat = b0
      for j in range(len(XtestMod)):
        yhat += XtestMod[j][i] * b1s[j]
      yhat = 1.0 / (1 + np.e ** -(yhat))
      if (yhat >= 0.5):
        if (y == 1): tp += 1
        else: fp += 1
      else:
        if (y == 0): tn += 1
        else: fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * (precision * recall) / (precision + recall))
    specificity = tn / (tn + fp)
    if (penalty != 0):
      nonzeroPenalties.append(penalty)
      precisions.append(precision)
      recalls.append(recall)
      f1s.append(f1)
      specificities.append(specificity)
      losses.append(rss)
    print("precision:", precision)
    print("recall:", recall)
    print("f1:", f1)
    print("specificity:", specificity)
    print()
  minIndex = 0
  minLoss = np.inf
  for i in range(len(losses)):
    if (losses[i] < minLoss):
      minIndex = i
      minLoss = losses[i]

  print("best penalty is " + str(nonzeroPenalties[minIndex]))
  print("precision:", precisions[minIndex])
  print("recall:", recalls[minIndex])
  print("f1:", f1s[minIndex])
  print("specificity:", specificities[minIndex])
  print()
  return

# main
def main():
  part1()
  part2()
  part3()


if __name__=="__main__":
  main()
