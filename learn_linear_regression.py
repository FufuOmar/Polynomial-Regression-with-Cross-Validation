# Import Libraries
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

D = np.loadtxt("D.dat", unpack=True)

D = D.T

m, n = D.shape

X = np.ones((m,n))

X[:,1:n] = D[:,0:(n-1)]
y = D[:,n-1]

# Create generalized linear regression object
regr = Ridge(alpha=0,fit_intercept=False,solver='cholesky')
# Train the model using the training sets
regr.fit(X, y)

np.savetxt("w.dat",regr.coef_)

