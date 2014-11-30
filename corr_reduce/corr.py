import pandas as pd
import numpy as np
import scipy
from sklearn.preprocessing import Imputer
from sklearn import mixture
import time
import sys
import os
import random
import itertools


log = open(os.path.splitext(sys.argv[0])[0]+'.log','wt')

print sys.argv

data = pd.read_csv(sys.argv[1])
fp = os.path.splitext(os.path.basename(sys.argv[1]))[0]
if len(sys.argv)>2:
  fp = (fp + '_'+ os.path.splitext(os.path.basename(sys.argv[2]))[0])
  print >> log, data.shape
  test = pd.read_csv(sys.argv[2])
  print >> log, test.shape
  data = pd.concat([data,test])

print >> log, data.shape

corr = data.corr()
print >> log, corr.shape

corr_filename = (os.path.splitext(os.path.basename(sys.argv[0]))[0]+'_'+ fp+".csv")
corr.to_csv(corr_filename)

log.close()


