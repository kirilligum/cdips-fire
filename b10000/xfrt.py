#!/bin/python
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import ExtraTreesRegressor
#from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
#import random
import sys
import os


log = open(os.path.splitext(sys.argv[0])[0]+'.log','wt')

data_train_in = pd.read_csv(sys.argv[1])
data_test_in = pd.read_csv(sys.argv[2])
print >> log,  data_train_in.shape, "  ", data_test_in.shape

start = time.clock()
target_train = data_train_in['target']
#print >> log, target_train
data_train = data_train_in.drop(['id','target'],axis=1)
if 'target' in data_test_in:
  target_test = data_test_in['target']
  data_test = data_test_in.drop(['id','target'],axis=1)
else:
  data_test = data_test_in.drop('id',axis=1)

xfr = ExtraTreesRegressor(n_estimators =1000,n_jobs=-1,max_depth=7)
xfr = xfr.fit(data_train,target_train)

out_filename = (os.path.splitext(os.path.basename(sys.argv[0]))[0]+os.path.splitext(os.path.basename(sys.argv[1]))[0]+"_xfr")
with open(out_filename+'_model.txt','wt') as f:
  print >> f, xfr
with open(out_filename+'_estimators_.txt','wt') as f:
  #f.write(xfr.estimators_)
  print >> f, xfr.estimators_
np.savetxt(out_filename+'_feature_importances_.txt',xfr.feature_importances_)
print data_train.columns.shape,xfr.feature_importances_.shape
with open(out_filename+'_fimp.txt','wt') as f:
  for feat,imp in zip(data_train.columns,xfr.feature_importances_):
    print >>f,"%s,%g"%(feat,imp)
#with open(out_filename+'_feature_importances_.txt','wt') as f:
  #print >> f, xfr.feature_importances_
#with open('oob_score_.txt','wt') as f:
  #print >> f, xfr.oob_score_
#with open('oob_prediction_.txt','wt') as f:
  #print >> f, xfr.oob_prediction_

transformed_train = xfr.transform(data_train)
transformed_test = xfr.transform(data_test)
end = time.clock()
print >> log, "time = ", end-start


suffix = '_tr.csv'
train_filename = (os.path.splitext(os.path.basename(sys.argv[1]))[0]+suffix)
train = pd.DataFrame(transformed_train)
train = pd.concat([data_train_in.ix[:,'target'],train],axis=1)
train = pd.concat([data_train_in.ix[:,'id'],train],axis=1)
train.to_csv(train_filename,index=0)
test_filename = (os.path.splitext(os.path.basename(sys.argv[2]))[0]+suffix)
test = pd.DataFrame(transformed_test)
if 'target' in data_test_in:
  test = pd.concat([data_test_in.ix[:,'target'],test],axis=1)
test = pd.concat([data_test_in.ix[:,'id'],test],axis=1)
test.to_csv(test_filename,index=0)

log.close()
