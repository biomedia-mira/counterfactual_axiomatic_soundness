import numpy as np

a = np.random.normal(loc=0.0, scale=1.0, size=(500, 1))
b = a ** 2
c = 3 * a
x = np.concatenate((a, b, c), axis=1)

# C^R_n_r means combinations with replacement of r elements from a set of n total elements

# mu_1: 3 moments (mean)
mu_1 = np.mean(x, axis=0, keepdims=True)
# mu_2: C^R_3_2 = 6 moments (tri of covariance matrix)
mu_2 = 1 / (x.shape[0] - 1) * np.transpose(x - mu_1) @ (x - mu_1) # np.cov(x, rowvar=False)
# mu_3: C^R_3_3 = 10 moments
