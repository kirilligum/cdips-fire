#!/bin/python
import pandas as pd
#import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor
#from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from gini import normalized_weighted_gini
#import random
import sys
import os


log = open(os.path.splitext(sys.argv[0])[0]+'.log','wt')

data_train = pd.read_csv(sys.argv[1])
data_test = pd.read_csv(sys.argv[2])
print >> log,  data_train.shape, "  ", data_test.shape

start = time.clock()
#print >> log, data_train.columns
target_train = data_train['target']
#print >> log, target_train
data_train.drop('target',axis=1,inplace=True)
if 'target' in data_test:
  target_test = data_test['target']
  data_test.drop('target',axis=1,inplace=True)

rfr = RandomForestRegressor(n_jobs=-1)
rfr = rfr.fit(data_train,target_train)
predict_loc_regres = rfr.predict(data_test)
if 'target' in data_test:
  score = rfr.score(data_test,target_test)
  gn = normalized_weighted_gini(target_test,predict_loc_regres,data_test.var11)
end = time.clock()

#outdf = pd.DataFrame([data_test.ix[:,'id']])
if 'target' in data_test:
  outdf = pd.concat([data_test.ix[:,'id'].astype(int),pd.DataFrame(predict_loc_regres,columns=['predicted']),target_test],axis=1)
else:
  outdf = pd.concat([data_test.ix[:,'id'].astype(int),pd.DataFrame(predict_loc_regres,columns=['predicted'])],axis=1)

out_filename = (os.path.splitext(os.path.basename(sys.argv[1]))[0]+"_predict.csv")
outdf.to_csv(out_filename,index=0)
print out_filename, score , gn

log.close()
