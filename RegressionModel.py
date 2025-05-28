import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 3, 4, 5, 6])
N = np.array([2.1, 3.4, 3.9, 4.7, 9.7])

plt.plot(X,Y, label = 'Line 1', color='blue', marker = 'o')

def statmodel():
    plt.xlabel('X-axis')
    plt.xlabel('Y-axis')
    plt.title('Simple Line Plot')
    plt.legend()
    plt.show()

def myfunc(slope, intercept):
  return (slope * X) + intercept

def regressionmodel():
    plt.scatter(X,Y)
    plt.show()
    slope, intercept, r, p, std_err = stats.linregress(X, Y)
    mymodel = list(map(myfunc(slope, intercept), X))
    plt.scatter(X, Y)
    plt.plot(X, mymodel)
    plt.show()

def msecalc(act,pred):
    return np.mean(np.square(np.subtract(act, pred)))

statmodel()
regressionmodel()
print("The MSE is: ", msecalc(Y,N))

