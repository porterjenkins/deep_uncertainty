import numpy as np
from scipy.special import factorial
import math
import matplotlib.pyplot as plt

def get_kappa(lam, gamma):
    first =(1 - gamma) / (12*gamma*lam)
    second = 1 + 1/(gamma*lam)
    return 1 + first * second

def pmf(y, lam, gam):
    #kappa = get_kappa(lam, gam)
    #first = kappa * np.power(gam, 0.5) * np.exp(-gam*lam)
    second = (np.exp(-y) * np.power(y, y)) / (factorial(y))
    third = np.power(((math.e * lam) / y), lam*gam)
    return second * third



y = np.arange(0, 10)

density = pmf(y, lam=5, gam=1)

print(density)
plt.plot(y, density, marker='o', linestyle='--')
plt.show()