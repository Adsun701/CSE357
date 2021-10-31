import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import sys

def q1():
    d1 = np.array([])
    d2 = np.array([])
    trials = 1
    threshold = 0.01
    while 1:
        d1 = np.append(d1, np.random.randint(1,7))
        d2 = np.append(d2, np.random.randint(1,7))
        rollsd1, freqd1 = np.unique(d1, return_counts=True)
        rollsd2, freqd2 = np.unique(d2, return_counts=True)

        if len(freqd1)==6 and len(freqd2)==6:
            flag = False
            for ind, i in enumerate(freqd1):
                if abs(1/6 - i/trials) < threshold:
                    continue
                else:
                    flag = True
                    break

            for jnd, j in enumerate(freqd2):
                if abs(1/6 - j/trials) < 0.01:
                    continue
                else:
                    flag = True
                    break
            
            if not flag:
                break
        
        trials+=1

    d1 = d1.astype(int)
    d2 = d2.astype(int)
    d3 = d1 + d2
    print("No. of trials : " + str(trials))
    print("\n")

    rollsd1, freqd1 = np.unique(d1, return_counts=True)
    rollsd2, freqd2 = np.unique(d2, return_counts=True)
    rollsd3, freqd3 = np.unique(d3, return_counts=True)



    print("####### Option a ######")
    print("Probability of D1=6 : " + str(freqd1[5]/trials))
    print("Probability of D1+D2=5 : " + str(freqd3[np.where(rollsd3 == 5)[0]][0]/trials))
    d1indices = np.where(d1 == 6)[0]
    d2indices = np.where(rollsd2 == -1)[0]
    a = 0
    b = 0
    if len(d2indices) == 0:
        print("Probability of D1=6 & D1+D2=5 : 0")
        a=0
    else:
        print("Probability of D1=6 & D1+D2=5 : " + str(len(np.where(d3[d1indices] == 5)[0])/trials))
        a=len(np.where(d3[d1indices] == 5)[0])/trials
    
    print("Probability of D1=6 * D1+D2=5 : " + str((freqd1[5]/trials) * (freqd3[np.where(rollsd3 == 5)[0]][0]/trials)))
    b=(freqd1[5]/trials) * (freqd3[np.where(rollsd3 == 5)[0]][0]/trials)
    if abs(a-b) < threshold:
        print("These two events are independent")
    else:
        print("These two events are not independent")
    print("\n")



    print("####### Option b ######")
    print("Probability of D1=5 : " + str(freqd1[4]/trials))
    print("Probability of D1+D2=6 : " + str(freqd3[np.where(rollsd3 == 6)[0]][0]/trials))
    d1indices = np.where(d1 == 5)[0]
    d2indices = np.where(rollsd2 == 1)[0]
    a = 0
    b = 0
    if len(d2indices) == 0:
        print("Probability of D1=5 & D1+D2=6 : 0")
        a=0
    else:
        print("Probability of D1=5 & D1+D2=6 : " + str(len(np.where(d3[d1indices] == 6)[0])/trials))
        a=len(np.where(d3[d1indices] == 6)[0])/trials
    print("Probability of D1=5 * D1+D2=6 : " + str((freqd1[4]/trials) * (freqd3[np.where(rollsd3 == 6)[0]][0]/trials)))
    b = (freqd1[4]/trials) * (freqd3[np.where(rollsd3 == 6)[0]][0]/trials)
    if abs(a-b) < threshold:
        print("These two events are independent")
    else:
        print("These two events are not independent")
    print("\n")



    print("####### Option c ######")
    print("Probability of D1=4 : " + str(freqd1[3]/trials))
    print("Probability of D1+D2=7 : " + str(freqd3[np.where(rollsd3 == 7)[0]][0]/trials))
    d1indices = np.where(d1 == 4)[0]
    d2indices = np.where(rollsd2 == 3)[0]
    a = 0
    b = 0
    if len(d2indices) == 0:
        print("Probability of D1=4 & D1+D2=7 : 0")
        a=0
    else:
        print("Probability of D1=4 & D1+D2=7 : " + str(len(np.where(d3[d1indices] == 7)[0])/trials))
        a= len(np.where(d3[d1indices] == 7)[0])/trials
    print("Probability of D1=4 * D1+D2=7 : " + str((freqd1[3]/trials) * (freqd3[np.where(rollsd3 == 7)[0]][0]/trials)))
    b=(freqd1[3]/trials) * (freqd3[np.where(rollsd3 == 7)[0]][0]/trials)
    if abs(a-b) < threshold:
        print("These two events are independent")
    else:
        print("These two events are not independent")

def q2():
    #### a ####
    print("##### a #####")
    mu = 42
    sigma = 8.5
    x= np.linspace(mu - 3*sigma, mu + 3*sigma, 100) 
    plt.title("PDF and CDF")
    plt.plot(x, ss.norm.pdf(x, mu, sigma), c="blue", label="pdf")
    plt.plot(x, ss.norm.cdf(x, mu, sigma), c="red", label="cdf")
    plt.legend()
    plt.show()

    #### b ####
    print("##### b #####")
    probability_to_die = (ss.norm.cdf(39.5, loc=mu, scale=sigma) - ss.norm.cdf(34, loc=mu, scale=sigma))/(1-ss.norm.cdf(34, loc=mu, scale=sigma))
    probability_to_die_2 = (ss.norm.cdf(39.5, loc=mu, scale=sigma) - ss.norm.cdf(34, loc=mu, scale=sigma))
    print("Probability to die before text recieved Way 1: " + str(probability_to_die))
    print("Probability to die before text recieved Way 2: " + str(probability_to_die_2))
    print("Density at the time the text recieved : " + str(ss.norm.pdf(39.5, loc=mu, scale=sigma)))

    #### c ####
    print("##### c #####")
    conf_95 = (1 - ss.norm.cdf(34, loc=mu, scale=sigma)) * 0.05
    hr = ss.norm.ppf(ss.norm.cdf(34, loc=mu, scale=sigma) + conf_95, loc=mu, scale=sigma)
    print("The time the text should be recieved from now is : " + 
    str(hr- 34))
    print("The time the text should be recieved from now is hr: " + 
    str(hr))

    #### d ####
    print("##### d #####")
    print("The probability to die while hanging out is Way 1: " + 
    str((ss.norm.cdf(42.5, loc=mu, scale=sigma) - ss.norm.cdf(40, loc=mu, scale=sigma))/(ss.norm.cdf(42.5, loc=mu, scale=sigma)- ss.norm.cdf(hr, loc=mu, scale=sigma))))
    print("The probability to die while hanging out is Way 2: " + 
    str((ss.norm.cdf(42.5, loc=mu, scale=sigma) - ss.norm.cdf(40, loc=mu, scale=sigma))/(1- ss.norm.cdf(hr, loc=mu, scale=sigma))))
    print("The probability to die while hanging out is Way 23: " + 
    str((ss.norm.cdf(42.5, loc=mu, scale=sigma) - ss.norm.cdf(40, loc=mu, scale=sigma))))
    plt.title("CDF between the start and end of hanging out")
    plt.plot(np.linspace(40, 42.5, 100), ss.norm.cdf(np.linspace(40, 42.5, 100), loc=mu, scale=sigma))
    plt.show()

def randKumaraswamy(size = 100, a = 2, b = 2):
    # References
    # https://www.calculushowto.com/inverse-distribution-function-point-quantile/
    # http://www.ntrand.com/kumaraswamy-distribution/
    # returns a list of size of random numbers drawn from a Kumaraswamy(a, b) distribution
    dist = [pow((1 - pow(1-x, 1/b)), 1/a) for x in np.random.random_sample(size)]
    return dist

def q3():
    r = randKumaraswamy()
    kde = ss.gaussian_kde(r)
    #For each sample from kumaraswamy distribution the kde values are plotted.
    plt.title("Guassian KDE for 100 values from Kumaraswamy distribution")
    plt.xlabel("samples from kumaraswamy dist")
    plt.ylabel("guassian kde values")
    plt.scatter(r, kde.evaluate(r))
    plt.show()

if __name__ == "__main__":
    sys.stdout = open('a1_part2_OUTPUT.txt', 'w')
    print("PART 2 - Q1")
    print("\n")
    q1()
    print("\n")
    print("PART 2 - Q2")
    print("\n")
    q2()
    print("\n")
    print("PART 2 - Q3")
    print("\n")
    q3()