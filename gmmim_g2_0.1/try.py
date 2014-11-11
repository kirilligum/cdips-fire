import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
import time
import sys
import os
import random
import itertools

#items =2
#x = np.array([i*1e300 for i in xrange(items)])
#mean = np.array([i*1e-10 for i in xrange(items)])
##cov = np.array([i*1e-10 for i in xrange(items*items)]).reshape(items,items)
##cov = np.random.rand(items,items)
#scale = 1e300
#cov = [[2.0*scale,0.3*scale],[0.3*scale,0.5*scale]]
##cov=cov*cov.transpose()
##pref = multivariate_normal.rvs()
x = 0
mean = [0.09090909]
cov = [[ 0.00191]]
print "x=",x
print "mean = ",mean
print "cov =",cov
pref = multivariate_normal.pdf(x,mean,cov)
print pref


