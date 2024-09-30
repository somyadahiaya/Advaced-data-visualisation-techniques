#Somya Dahiaya  #

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.stats import binom
from scipy.stats import gamma
from scipy.stats import norm

data = np.loadtxt("3.data")

# task A

mu_1 = np.mean(data)

mu_2 = np.mean(data ** 2)

variance = mu_2 - (mu_1 ** 2)

print(mu_1)
print(mu_2)
print(variance)

# task B

plt.hist(data, bins = 120, edgecolor = 'black', density = True, alpha = 0.5, label = 'data')

plt.title('Histogram of the dataset')
plt.xlabel('Value')
plt.ylabel('Probability')

plt.savefig('3b.png')

plt.show()

#task C

def moments_binom(params):
    n, p = params
    mu_1_bin = n * p
    mu_2_bin = (n * p * (1 - p)) + ((n * p) ** 2)
    return [mu_1_bin - mu_1, mu_2_bin - mu_2]

n_p_start = [10, 0.5]

n, p = fsolve(moments_binom, n_p_start)

diff_n_floor = n - int(n)
diff_n_ceil = round(n) - n

if(diff_n_floor >= diff_n_ceil):
    n1 = round(n)
else:
    n1 = int(n)

print(f"Approx n*: {n}")
print(f"Approx p*: {p:.4f}")

x_binom = np.linspace(0, n1 + 1, n1 + 2)

pmf_values = binom.pmf(x_binom, n1, p)

plt.plot(x_binom, pmf_values, label=f'binom(n={n1}, p={p:.2f})', color = 'red')

plt.title('Best binomial distribution approximation to the true distribution')
plt.xlabel('x')
plt.ylabel('p(x)')
plt.legend()

plt.savefig('3c.png')

plt.show()

#task D

def moments_gamma(params):
    k, theta = params
    mu_1_gamma = k * theta
    mu_2_gamma = k * (theta ** 2)
    return [mu_1_gamma - mu_1, mu_2_gamma - variance]

k_theta_start = [5,5]

k, theta = fsolve(moments_gamma, k_theta_start)

print(f"Approx k*: {k:.4f}")
print(f"Approx θ*: {theta:.4f}")

x_gamma = np.linspace(0, 20, 1000)
pdf_values = gamma.pdf(x_gamma, a=k, scale=theta)

plt.clf()
plt.hist(data, bins = 120, edgecolor = 'black', density = True, alpha = 0.5, label = 'data')
plt.plot(x_gamma, pdf_values, label=f'Gamma(k={k:.2f}, θ={theta:.2f})', color='red')
plt.title('Best Gamma distribution approximation to the true distribution')
plt.xlabel('x')
plt.ylabel('Probability density')
plt.legend()

plt.savefig('3d.png')
plt.show()

#Task E

rounded_data = np.round(data).astype(int)

rounded_data = rounded_data[(rounded_data >= 0) & (rounded_data <= n1)]

log_likelihood_binom = np.log(binom.pmf(rounded_data, n1, p))

average_log_likelihood_binom = np.mean(log_likelihood_binom)

log_likelihood_gamma = np.log(gamma.pdf(data, a = k, scale = theta))

average_log_likelihood_gamma = np.mean(log_likelihood_gamma)

print(f"Average log-likelihood for the Binomial distribution: {average_log_likelihood_binom:.4f}")
print(f"Average log-likelihood for the Gamma distribution: {average_log_likelihood_gamma:.4f}")

if average_log_likelihood_gamma > average_log_likelihood_binom:
    print("The Gamma distribution is a better fit.")
else:
    print("The Binomial distribution is a better fit.")
    
#Task F

mu_3 = np.mean(data ** 3)
mu_4 = np.mean(data ** 4)

print(f"Third moment (mu_3): {mu_3:.4f}")
print(f"Fourth moment (mu_4): {mu_4:.4f}")

def moments_gmm(params):
    mu1, p1, mu2, p2 = params
    sigma1 = sigma2 = 1
    
    mu1_gmm = p1 * mu1 + p2 * mu2
    mu2_gmm = p1 * (sigma1 + mu1 ** 2) + p2 * (sigma2 + mu2 ** 2)
    mu3_gmm = p1 * (mu1 ** 3 + 3 * mu1 * sigma1) + p2 * (mu2 ** 3 + 3 * mu2 * sigma2)
    mu4_gmm = p1 * (mu1 ** 4 + 6 * mu1 ** 2 * sigma1 + 3 * sigma1 ** 2) + p2 * (mu2 ** 4 + 6 * mu2 ** 2 * sigma2 + 3 * sigma2 ** 2)
    
    return [mu1_gmm - mu_1, mu2_gmm - mu_2, mu3_gmm - mu_3, mu4_gmm - mu_4]

inital_values = [mu_1 - 1, 0.5, mu_1 + 1, 0.5]
mu1, p1, mu2, p2 = fsolve(moments_gmm, inital_values)

print(f"Approx parameters: mu1 = {mu1:.4f}, p1 = {p1:.4f}, mu2 = {mu2:.4f}, p2 = {p2:.4f}")

x_gmm = np.linspace(0, 20, 1000)
pdf_gmm = p1 * norm.pdf(x_gmm, mu1, 1) + p2 * norm.pdf(x_gmm, mu2, 1)

plt.clf()
plt.hist(data, bins=120, edgecolor='black', density=True, alpha=0.5, label='data')
plt.plot(x_gmm, pdf_gmm, label=f'GMM({p1:.2f}N({mu1:.2f},1), {p2:.2f}N({mu2:.2f},1))', color='red')
plt.title('Best two-component unit-variance GMM approximation to the true distribution')
plt.xlabel('x')
plt.ylabel('p(x)')
plt.legend()
plt.savefig('3f.png')
plt.show()

log_likelihood_gmm = np.log(p1 * norm.pdf(data, mu1, 1) + p2 * norm.pdf(data, mu2, 1))
average_log_likelihood_gmm = np.mean(log_likelihood_gmm)

print(f"Average log-likelihood for the GMM: {average_log_likelihood_gmm:.4f}")

print("The Binomial distribution is a better approimation than the GMM and Gamma distribution.")