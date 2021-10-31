import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import csv

def schoolBinomialHypTest(null_school, test_school, alpha, school_data):
  #null_school: the school representing the null hypothesis distribution
  #test_school: the school representing the observed count
  #alpha: significance level
  #school_data: the school name, case counts, and population
  with open(school_data, 'r') as csv_file:
    # create a csv reader.
    csv_reader = csv.reader(csv_file)
    
    row_null = []
    row_test = []
    
    college_col = -1
    population_col = -1
    cases_col = -1
    
    # for first row, check index of college column, population column, and cases column.
    row1 = next(csv_reader)
    for i in range(len(row1)):
      if row1[i] == "college": college_col = i
      elif row1[i] == "population": population_col = i
      elif row1[i] == "cases": cases_col = i
    
    # if any index is -1, a required column was not found. exit if so.
    if (college_col < 0 or population_col < 0 or cases_col < 0):
      print("Error: inputed csv file does not contain \"college\", \"population\", or \"cases\" column.")
      return -1, "error"
    
    # loop until we reach null school or test school.
    for row in csv_reader:
      if (row[college_col] == null_school):
        row_null = row
      elif (row[college_col] == test_school):
        row_test = row
      # if we have gotten both rows for null and test schools, break.
      if row_null and row_test: break
    
    # if null school or test school not found, exit.
    if (not row_null):
      print("Error: null school not found.")
      return -1, "error"
    elif (not row_test):
      print("Error: test school not found.")
      return -1, "error"
    
    # get populations for test and null schools.
    n1 = float(row_test[population_col])
    n2 = float(row_null[population_col])
    
    # get numbers of those infected in each school.
    c1 = float(row_test[cases_col])
    c2 = float(row_null[cases_col])
    
    # get the cdf of the binomial distribution of
    # where x is number of cases of test school minus 1,
    # n is population of test school, and p is
    # number of cases in null school divided by the total
    # population of null school. subtract the resulting value from 1.
    p = 1 - ss.binom.cdf(c1 - 1, n1, c2 / n2)
    if p < alpha: decision = "reject"
    else: decision = "accept"
    
  return p, decision

def getListOfSchools(school_data):
  with open(school_data, 'r') as csv_file:
    # create a csv reader.
    csv_reader = csv.reader(csv_file)
    row1 = next(csv_reader)
    college_col = -1
    for i in range(len(row1)):
      if row1[i] == "college": college_col = i
    if college_col < 0: return []
    school_list = []
    for row in csv_reader:
      school_list.append(row[college_col])
    return school_list

def getInfectionRate(school, school_data):
  with open(school_data, 'r') as csv_file:
    # create a csv reader.
    csv_reader = csv.reader(csv_file)
    row1 = next(csv_reader)
    college_col = -1
    population_col = -1
    cases_col = -1
    for i in range(len(row1)):
      if row1[i] == "college": college_col = i
      elif row1[i] == "population": population_col = i
      elif row1[i] == "cases": cases_col = i
    if college_col < 0 or population_col < 0 or cases_col < 0: return -1
    for row in csv_reader:
      if (row[college_col] == school):
        return float(row[cases_col]) / float(row[population_col])
    return -1

def maxLengthOfString(arr):
  maxLen = 0
  for s in arr:
    if (len(s) > maxLen): maxLen = len(s)
  return maxLen
    

# compares null_school to all other schools in school_data.
def schoolBinomialMultipleHypTest(null_school, alpha, school_data):
  #null_school: the school representing the null hypothesis distribution
  #alpha: significance level
  #school_data: the school name, case counts, and population
  school_list = getListOfSchools(school_data)
  maxLen = maxLengthOfString(school_list)
  numTestSchools = len(school_list) - 1
  rate_null = getInfectionRate(null_school, school_data)
  for test_school in school_list:
    if (test_school == null_school): continue
    p, decision = schoolBinomialHypTest(null_school, test_school, alpha, school_data)
    rate_test = getInfectionRate(test_school, school_data)
    # print name of test school, difference in proportions of cases
    # between test school and null school, original p-value and
    # bonferroni corrected p-value.
    print(test_school.ljust(maxLen + 1) + ",\t diff: " +
      str(round(rate_test - rate_null, 6)) + ", orig p: " +
      str(p) + ", corrected p: " + str(p*numTestSchools))
  


# returns the degrees of freedom for a two-sample t-test
def degreesOfFreedom(v1, v2, n1, n2):
  return (((v1 / n1 + v2 / n2) ** 2) /
    ((v1 / n1)**2 / (n1 - 1) + (v2 / n2)**2 / (n2 - 1)))

# returns the variance given the number of 1s and 0s.
def variance(num0s, num1s):
  mu = num1s / (num0s + num1s)
  return (num0s * (0 - mu) ** 2 + num1s * (1 - mu) ** 2) / (num0s + num1s - 1)

def schoolNormalHypTest(null_school, test_school, alpha, school_data):
  #null_state: the school representing the null hypothesis distribution
  #test_state: the school representing the observed count
  #alpha: significance level
  #school_data: the school name, case counts, and population
  with open(school_data, 'r') as csv_file:
    # create a csv reader.
    csv_reader = csv.reader(csv_file)
    
    row_null = []
    row_test = []
    
    college_col = -1
    population_col = -1
    cases_col = -1
    
    # for first row, check index of college column, population column, and cases_2021 column.
    row1 = next(csv_reader)
    for i in range(len(row1)):
      if row1[i] == "college": college_col = i
      elif row1[i] == "population": population_col = i
      elif row1[i] == "cases": cases_col = i
    
    # if any index is -1, a required column was not found. exit if so.
    if (college_col < 0 or population_col < 0 or cases_col < 0):
      print("Error: inputed csv file does not contain \"college\", \"population\", or \"cases\" column.")
      sys.exit(1)
    
    # loop until we reach null school or test school.
    for row in csv_reader:
      if (row[college_col] == null_school):
        row_null = row
      elif (row[college_col] == test_school):
        row_test = row
      # if we have gotten both rows for null and test schools, break.
      if row_null and row_test: break
    
    # if null school or test school not found, exit.
    if (not row_null):
      print("Error: null school not found.")
      sys.exit(1)
    elif (not row_test):
      print("Error: test school not found.")
      sys.exit(1)
    
    # get populations for test and null schools.
    pop_test = float(row_test[population_col])
    pop_null = float(row_null[population_col])
    
    # get numbers of those infected in each school.
    cases_test = float(row_test[cases_col])
    cases_null = float(row_null[cases_col])
    
    # get means.
    mu_test = cases_test / pop_test
    mu_null = cases_null / pop_null
    
    # get variances.
    variance_test = variance(pop_test - cases_test, cases_test)
    variance_null = variance(pop_null - cases_null, cases_null)
    
    # get degrees of freedom.
    degrees = degreesOfFreedom(variance_test, variance_null, pop_test, pop_null)
    
    # calculate t-statistic.
    tStat = (mu_test - mu_null) / (variance_test / pop_test + variance_null / pop_null) ** 0.5
    
    # calculate p-value using tStat and cdf.
    p = 1 - ss.t.cdf(tStat, degrees)
    if p < alpha: decision = "reject"
    else: decision = "accept"
    
  return p, decision

# for q4().
def std_err_ci():
  school_data = "ny_colleges.csv"
  school_list = {}
  school_name = ""
  alpha = 0.05
  mu = 0
  muPrime = 1
  v = 0
  std_err = 0
  z = ss.norm.ppf(1 - alpha / 2.0)
  with open(school_data, 'r') as csv_file:
    # create a csv reader.
    csv_reader = csv.reader(csv_file)
    
    row_null = []
    row_test = []
    
    college_col = -1
    population_col = -1
    cases_col = -1
    
    # for first row, check index of college column, population column, and cases column.
    row1 = next(csv_reader)
    for i in range(len(row1)):
      if row1[i] == "college": college_col = i
      elif row1[i] == "population": population_col = i
      elif row1[i] == "cases": cases_col = i
    
    # if any index is -1, a required column was not found. exit if so.
    if (college_col < 0 or population_col < 0 or cases_col < 0):
      print("Error: inputed csv file does not contain \"college\", \"population\", or \"cases\" column.")
      return
    
    # read rest of csv into a list.
    data = list(csv_reader)
    for row in data:
      school_name = row[college_col]
      cases = float(row[cases_col])
      population = float(row[population_col])
      mu = cases / population
      muPrime = (population - cases) / population
      
      # prevent errors if cases are somehow greater than population
      if (muPrime < 0): muPrime = 0
      if (mu > 1): mu = 1
      
      v = variance(population - cases, cases)
      std_err = (v / population) ** 0.5
      lb_err = mu - z * std_err
      ub_err = mu + z * std_err
      
      school_list[school_name] = {}
      
      school_list[school_name]['mean'] = mu
      school_list[school_name]['lb_err'] = lb_err
      school_list[school_name]['ub_err'] = ub_err
      school_list[school_name]['population'] = population
    
    return school_list

# question 1. Compare SBU to New York University.
def q1():
  null_school = "Stony Brook University"
  test_school = "New York University"
  alpha = 0.05
  prob1, decision1 = schoolBinomialHypTest(null_school, test_school, alpha, "ny_colleges.csv")
  print(test_school + ", decision: " + decision1 + "; p: " + str(prob1))
  print("\n")

# question 2. Comparing SBU to Many New York Schools.
# Use Bonferroni correction for p-values.
def q2():
  null_school = "Stony Brook University"
  alpha = 0.05
  schoolBinomialMultipleHypTest(null_school, 0.05, "ny_colleges.csv")
  print("\n")

# question 3. Repeat question 1 with a normal distribution
# and t-test instead of a binomial.
def q3():
  null_school = "Stony Brook University"
  test_school = "New York University"
  alpha = 0.05
  prob1, decision1 = schoolNormalHypTest(null_school, test_school, alpha, "ny_colleges.csv")
  print(test_school + ", decision: " + decision1 + "; p: " + str(prob1))
  print("\n")

# question 4. Standard Error based 95% CI.
def q4():
  return std_err_ci()

# question 5. Bootstrapped 95% CI.
def q5(school_list):
  alpha = 0.05
  iters = 1000
  resample_len = 0
  
  # set max length of name of school.
  maxLen = 0
  
  for school_name in school_list:
    if (len(school_name) > maxLen): maxLen = len(school_name)
    school = school_list[school_name]
    resample_len = int(school["population"])
    all_means = []

    for i in range(iters):
      resample = np.random.choice(2, size=resample_len,
        replace=True, p=[1.0 - school["mean"], school["mean"]])
      all_means.append(resample.mean())
    all_means = sorted(all_means)
    lb_btstrp = all_means[int(0.5*alpha*iters)]
    ub_btstrp = all_means[-int(0.5*alpha*iters)]
    school["lb_btstrp"] = lb_btstrp
    school["ub_btstrp"] = ub_btstrp
    
  for school_name in school_list:
    school = school_list[school_name]
    print(school_name.ljust(maxLen + 1) + ", " + str(np.around(school["mean"], 4)) +
      ", stderr CI: [" + str(np.around(school["lb_err"], 4)) + ", "
      + str(np.around(school["ub_err"], 4)) + "], bootstrapped CI: ["
      + str(np.around(school["lb_btstrp"], 4)) + ", " +
      str(np.around(school["ub_btstrp"], 4)) + "]")
  
print("Question 1:")
q1()

print("Question 2:")
q2()

print("Question 3:")
q3()

print("Question 4:")
school_list = q4()

print("Question 5:")
q5(school_list)
