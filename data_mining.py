from sklearn import datasets
import numpy as np
import math
from sklearn import preprocessing
import scipy
from scipy.stats import norm
import matplotlib.pyplot as plt
from __future__ import division

iris = datasets.load_iris()

sepal_length = iris.data[:,0]

# sample range
sepal_range = max(sepal_length) - min(sepal_length)

# sample IQR
def Cdf(t, x):
    count = 0.0
    for value in t:
        if value <= x:
            count += 1.0

    prob = count / len(t)
    return prob

def ICdf(t, x):
    for value in sorted(t):
        if Cdf(t, value) >= x:
            return value
IQR = ICdf(sepal_length, 0.75) - ICdf(sepal_length, 0.25)

# sample variance
np.var(sepal_length)
np.std(sepal_length)

# bivariate

# mean and covariance
X12 = iris.data[:,0:2]
# column-wise mean
np.mean(X12, axis=0)
# covariance matrix
covmat = np.round(np.cov(np.transpose(X12)), 3)
# correlation
covmat[0,1] / math.sqrt(covmat[0,0] * covmat[1,1])
cormat = np.round(np.corrcoef(np.transpose(X12)), 3)
# cosine angle
cosang = math.degrees(math.acos(cormat[0,1]))
# matrix total variance (trace)
np.matrix.trace(covmat)
# matrix determinant
np.linalg.det(covmat)
# eigenvalues and eigenvectors of covariance matrix
eigvals, eigvects = np.linalg.eig(covmat)
# rotation angle for multivariate normal (?)
np.round(math.degrees(math.acos(eigvects[0,0])), 3)

# multivariate
# mean
np.round(np.mean(iris.data, axis=0), 3)
# covariance matrix
covmat_multi = np.round(np.cov(np.transpose(iris.data)), 3)
# total variance (trace)
np.matrix.trace(covmat_multi)
# generalized variance (determinant)
'%.3e' % np.linalg.det(covmat_multi)

# data normalization

# range normalization
# = (x - min) / (max(x) - min(x))
range_scaler = preprocessing.MinMaxScaler()
iris_minmax = range_scaler.fit_transform(iris.data)
# standard score normalization: each value replaced by its z-score
iris_zscore = scipy.stats.zscore(iris.data)

# normal distribution
loc = 0; scale = 1
fig, ax = plt.subplots(1, 1)
mean, var, skew, kurt = norm.stats(moments='mvsk', loc=loc, scale=scale)
x = np.linspace(norm.ppf(0.01, loc=loc, scale=scale), norm.ppf(0.99, loc=loc, scale=scale), 100)
ax.plot(x, norm.pdf(x, loc=loc, scale=scale), 'r-', lw=5, alpha=0.6, label='norm pdf')
loc = 0; scale = 2
mean, var, skew, kurt = norm.stats(moments='mvsk', loc=loc, scale=scale)
x = np.linspace(norm.ppf(0.01, loc=loc, scale=scale), norm.ppf(0.99, loc=loc, scale=scale), 100)
ax.plot(x, norm.pdf(x, loc=loc, scale=scale), '-', lw=5, alpha=0.6, label='norm pdf')

# categorical attributes
sepal_length_categ = np.array(["Long" if x >= 7 else "Short" for x in sepal_length])
n = len(sepal_length_categ)
p = sum(sepal_length_categ == "Long") / n
sepal_length_categ_mean = p
sepal_length_categ_var = p * (1 - p)
# number of occurrences
n_long_sepal_length = n * p # not surprising, because p was derived from data
var_n_long_sepal_length = n * p * (1 - p) # more interesting, the variance in n of occurrences
# so, 95% CI of mean (mean +- 2 std)
CI_n_long_sepal_length = [n_long_sepal_length - (math.sqrt(var_n_long_sepal_length) * x) for x in (2, -2)]






