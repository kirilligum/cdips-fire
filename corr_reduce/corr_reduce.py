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

data_train_in = pd.read_csv(sys.argv[1])
data_test_in = pd.read_csv(sys.argv[2])
train = data_train_in.drop('target',axis=1)
if 'target' in data_test_in:
  test = data_test_in.drop('target',axis=1)
else:
  test = data_test_in
data = pd.concat([train,test])
print >> log,  train.shape, "  ", test.shape ,"  " , data.shape

corr = data.corr()
print corr.shape
print corr

corr_filename = (os.path.splitext(os.path.basename(sys.argv[1]))[0]+"_corr.csv")
corr.to_csv(corr_filename,index=0)

print corr_filename

#delcols = []

#current_cols = corr.columns[:1]

#while len(current_cols) > 0:
  #icol = current_cols[0]
  #correlated = corr[corr[icol]==1].index.tolist()
  #print corr[icol]
  #print 'corr is one',corr[corr[icol]==1]
  #print 'correlated',correlated
  #current_cols = current_cols.drop(current_cols[0])
  #print current_cols

  #for idx,row in corr.iterrows():
    #print type(row) , row.shape, corr[idx].name



log.close()


