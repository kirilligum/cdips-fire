#!/bin/python
import pandas as pd
#import numpy as np
import time
#from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
#from sklearn.metrics import accuracy_score
#import random
import sys
import os


log = open(os.path.splitext(sys.argv[0])[0]+'.log','wt')

data_train_in = pd.read_csv(sys.argv[1])
data_test_in = pd.read_csv(sys.argv[2])
print >> log,  data_train_in.shape, "  ", data_test_in.shape

start = time.clock()
#print >> log, data_train_in.columns
target_train = data_train_in['target']
#print >> log, target_train
data_train = data_train_in.drop(['id','target'],axis=1)
data_test = data_test_in.drop('id',axis=1)
if 'target' in data_test:
  target_test = data_test_in['target']
  data_test.drop('target',axis=1,inplace=True)

#xfr = ExtraTreesRegressor(n_estimators =1000,n_jobs=-1,max_depth=7)
rfr = RandomForestRegressor(n_estimators =1000,n_jobs=-1,max_depth=7)
rfr = rfr.fit(data_train,target_train)

with open('model.txt','wt') as f:
  print >> f, rfr
with open('estimators_.txt','wt') as f:
  #f.write(rfr.estimators_)
  print >> f, rfr.estimators_
with open('feature_importances_.txt','wt') as f:
  print >> f, rfr.feature_importances_

predtest = rfr.predict(data_test)
end = time.clock()

if 'target_test' in locals():
  target_test.columns = ['true_target']
  outdf = pd.concat([data_test_in.ix[:,'id'].astype(int),pd.DataFrame(predtest,columns=['target']),target_test],axis=1)
else:
  outdf = pd.concat([data_test_in.ix[:,'id'].astype(int),pd.DataFrame(predtest,columns=['target'])],axis=1)

out_filename = (os.path.splitext(os.path.basename(sys.argv[2]))[0]+"_rfr.csv")
outdf.to_csv(out_filename,index=0)
print out_filename

log.close()
