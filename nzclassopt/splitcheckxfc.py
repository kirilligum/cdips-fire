#!/bin/python
import pandas as pd
#import numpy as np
import time
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from gini import normalized_weighted_gini
#import random
import sys
import os

def nz_split(data,nsp):
  zd= data[data[target]==0]#data with targets=0
  nzd= data[data[target]!=0]#data with targets!=0
  splits =[] #create  subsets of data of almost equal size for zero targets
  for i in range(nsp):
    isplit =zd[i*len(zd)/nsp:(i+1)*len(zd)]
    isplit+=[nzd]
    splits +=[isplit]
  return splits

log = open(os.path.splitext(sys.argv[0])[0]+'.log','wt')

data_train = pd.read_csv(sys.argv[1])
splits= sys.argv[3]
print >> log,  data_train.shape, "  ",  "  splits = ", splits

trains = nz_split(data_train,splits)
xfc = ExtraTreesClassifier(n_estimators =10000,n_jobs=-1,max_depth=7)

for i in trains:
  target_train = i['target']
  data_train_nt = i.drop('target',axis=1)
  xfc = xfc.fit(data_train_nt,[0 if x==0 else 1 for x in target_train])
  predict_proba  = pd.concat([i,pd.DataFrame(xfc.predict(data_train_nt),columns=['predicted']), pd.DataFrame( xfc.predict_proba(data_train_nt),columns=['proba0','proba1'])],axis=1).sort(['target','proba0'],ascending=[False,False])

  #print "false positives", len(predcted_prob[(predcted_prob['target']==0)].any() and predcted_prob[(predcted_prob['predicted']==1)].any())
  print >> log, "false positives", len(predcted_prob[(predcted_prob['target']==0) & (predcted_prob['predicted']==1)])
  print >> log, "false negative", len(predcted_prob[(predcted_prob['target']!=0) & (predcted_prob['predicted']==0)])
  print >> log, "true positives", len(predcted_prob[(predcted_prob['target']!=0) & (predcted_prob['predicted']==1)])
  print >> log, "true negative", len(predcted_prob[(predcted_prob['target']==0) & (predcted_prob['predicted']==0)])

  print >> log, "nonzeros = ", len(target_train[target_train!=0])
  print >> log, "zeros = ", len(target_train[target_train==0])

  corrects += [xfc.predict(i)]

print corrects
ave = average(corrects)
print ave




start = time.clock()
#print >> log, data_train.columns
target_train = data_train['target']
#print >> log, target_train
data_train.drop('target',axis=1,inplace=True)
if 'target' in data_test:
  target_test = data_test['target']
  data_test.drop('target',axis=1,inplace=True)

xfr = ExtraTreesRegressor(n_estimators =1000,n_jobs=-1,max_depth=7)
xfr = xfr.fit(data_train,target_train)

with open('model.txt','wt') as f:
  print >> f, xfr
with open('estimators_.txt','wt') as f:
  #f.write(xfr.estimators_)
  print >> f, xfr.estimators_
with open('feature_importances_.txt','wt') as f:
  print >> f, xfr.feature_importances_
#with open('oob_score_.txt','wt') as f:
  #print >> f, xfr.oob_score_
#with open('oob_prediction_.txt','wt') as f:
  #print >> f, xfr.oob_prediction_

predict_loc_regres = xfr.predict(data_test)
if 'target_test' in locals():
  score = xfr.score(data_test,target_test)
  gn = normalized_weighted_gini(target_test,predict_loc_regres,data_test.var11)
end = time.clock()

#outdf = pd.DataFrame([data_test.ix[:,'id']])
if 'target_test' in locals():
  target_test.columns = ['true_target']
  outdf = pd.concat([data_test.ix[:,'id'].astype(int),pd.DataFrame(predict_loc_regres,columns=['target']),target_test],axis=1)
else:
  outdf = pd.concat([data_test.ix[:,'id'].astype(int),pd.DataFrame(predict_loc_regres,columns=['target'])],axis=1)

out_filename = (os.path.splitext(os.path.basename(sys.argv[1]))[0]+"_predict.csv")
outdf.to_csv(out_filename,index=0)
if 'target_test' in locals():
  print out_filename, score , gn
else:
  print out_filename

log.close()
