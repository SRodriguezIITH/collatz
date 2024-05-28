import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

n = np.array([3, 5, 7, 9])
r = np.array([22.6667, 44.2222, 124.5567, 204.0000])

def logistic(n, L, k, n0):
    return L / (1 + np.exp(-k * (n - n0)))

# Initial guesses for L, k, and n0 to reduce iterations
initial_guesses = [210, 0.5, 6]

params, covariance = curve_fit(logistic, n, r, p0=initial_guesses)
L, k, n0 = params

print(f"Estimated parameters:\nL = {L}\nk = {k}\nn0 = {n0}")

n_fit = np.linspace(2, 10, 100)
r_fit = logistic(n_fit, L, k, n0)

plt.scatter(n, r, label='Data')
plt.plot(n_fit, r_fit, label='Logistic Fit', color='red')
plt.xlabel('n')
plt.ylabel('r')
plt.legend()
plt.show()
