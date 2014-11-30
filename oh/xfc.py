#!/bin/python
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score
#from sklearn.metrics import r2_score
#from gini import normalized_weighted_gini
#import random
import sys
import os


log = open(os.path.splitext(sys.argv[0])[0]+'.log','wt')

data_train = pd.read_csv(sys.argv[1])
print >> log,  data_train.shape

start = time.clock()
#print >> log, data_train.columns
target_train = data_train['target']
#print >> log, target_train
data_train_zeros = data_train[data_train['target']==0]
data_train_zeros_nt = data_train_zeros.drop('target',axis=1)
data_train_nt = data_train.drop('target',axis=1)

#xfc = ExtraTreesClassifier(n_estimators =1000,n_jobs=-1,oob_score=True,max_depth=7)
xfc = ExtraTreesClassifier(n_estimators =10000,n_jobs=-1,max_depth=7)
#xfc = RandomForestClassifier (n_estimators =10000,n_jobs=-1,max_depth=7)
print >> log, target_train.shape
xfc = xfc.fit(data_train_nt,[0 if x==0 else 1 for x in target_train])

#with open('model.txt','wt') as f:
  #print >> f, xfr
#with open('estimators_.txt','wt') as f:
  ##f.write(xfr.estimators_)
  #print >> f, xfr.estimators_
#with open('feature_importances_.txt','wt') as f:
  #print >> f, xfr.feature_importances_
#with open('oob_score_.txt','wt') as f:
  #print >> f, xfr.oob_score_
#with open('oob_decision_function_.txt','wt') as f:
  #print >> f, xfr.oob_decision_function_


predcted_prob  = pd.concat([data_train,pd.DataFrame(xfc.predict(data_train_nt),columns=['predicted']), pd.DataFrame( xfc.predict_proba(data_train_nt),columns=['proba0','proba1'])],axis=1).sort(['target','proba0'],ascending=[False,False])

#print "false positives", len(predcted_prob[(predcted_prob['target']==0)].any() and predcted_prob[(predcted_prob['predicted']==1)].any())
print >> log, "false positives", len(predcted_prob[(predcted_prob['target']==0) & (predcted_prob['predicted']==1)])
print >> log, "false negative", len(predcted_prob[(predcted_prob['target']!=0) & (predcted_prob['predicted']==0)])
print >> log, "true positives", len(predcted_prob[(predcted_prob['target']!=0) & (predcted_prob['predicted']==1)])
print >> log, "true negative", len(predcted_prob[(predcted_prob['target']==0) & (predcted_prob['predicted']==0)])

print >> log, "nonzeros = ", len(target_train[target_train!=0])
print >> log, "zeros = ", len(target_train[target_train==0])

#print type(proba),proba.shape,proba
#np.savetxt("proba.csv",proba,delimiter=", ")
#log_proba = xfc.predict_log_proba(data_train_nt)


#if 'target_train' in locals():
  #score = xfc.score(data_train_nt,[0 if x is 0 else 1 for x in target_train])
end = time.clock()

#if 'target_train' in locals():
  #target_train.columns = ['true_target']
  #outdf = pd.concat([data_train_nt.ix[:,'id'].astype(int),pd.DataFrame(predict_loc_classifier,columns=['pred_target'])],axis=1)
  #outdf = pd.concat([outdf,pd.DataFrame(proba,columns=['proba0','proba1'])],axis=1)
  #outdf = pd.concat([outdf,pd.DataFrame(log_proba,columns=['log_proba'])],axis=1)
  #outdf = pd.concat([outdf,target_train],axis=1)
  #outdf = pd.concat([data_train_nt.ix[:,'id'].astype(int),pd.DataFrame(predict_loc_classifier,columns=['pred_target']),pd.DataFrame(proba,columns=['proba']),pd.DataFrame(log_proba,columns=['log_proba']),target_train],axis=1)
#else:
  #outdf = pd.concat([data_train_nt.ix[:,'id'].astype(int),pd.DataFrame(predict_loc_classifier,columns=['pred_target'])],axis=1)
  #outdf = pd.concat([data_train_nt.ix[:,'id'].astype(int),pd.DataFrame(predict_loc_classifier,columns=['target']),pd.DataFrame(proba,columns=['proba']),pd.DataFrame(log_proba,columns=['log_proba'])],axis=1)

out_filename = (os.path.splitext(os.path.basename(sys.argv[1]))[0]+"_prob.csv")
#predcted_prob.to_csv("predproba.csv",index=0)
predcted_prob.to_csv(out_filename,index=0)
#if 'target_train' in locals():
  #print out_filename, score
#else:
  #print out_filename

log.close()
