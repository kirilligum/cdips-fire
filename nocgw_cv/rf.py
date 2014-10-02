#!/bin/python
import pandas as pd
#import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor
#from sklearn.metrics import accuracy_score
#from gini import normalized_weighted_gini
#import random
import sys
import os


log = open(os.path.splitext(sys.argv[0])[0]+'.log','wt')

if not sys.stdin.isatty():
    inputfile = sys.stdin
    inputfile_prefix = open(inputfile).read()
    outputfile = sys.argv[0]
    if outputfile[-3:]=='.py':
        outputfile = outputfile[:-3]

elif len(sys.argv) > 1:
    inputfile = sys.argv[1]
    inputfile_prefix = open(inputfile).read()
    outputfile = sys.argv[0]
    if outputfile[-3:]=='.py':
        outputfile = outputfile[:-3]
    outputfile = outputfile + "_" + os.path.splitext(inputfile)[0]
    if len(sys.argv)>2:
        outputfile = sys.argv[2]
else :
    print >> log,  "no inputs"
    inputfile = 'train.csv'
    outputfile = sys.argv[0]
print >> log,  "input: ", inputfile
print >> log,  "output", outputfile
#data = pd.read_csv( inputfile)
#print >> log,  data.shape

print "input_info = ", inputfile_prefix
input_prefix, Null , folds = inputfile_prefix.rpartition('_f')
print "prefix = ", input_prefix
print "folds = ", folds

log.close()

for i in range(int(folds)):
  train_name = input_prefix+'_train_'+str(i)+'.csv'
  test_name = input_prefix+'_test_'+str(i)+'.csv'
  data_train = pd.read_csv(train_name)
  data_test = pd.read_csv(test_name)
  print train_name, data_train.shape, test_name, data_test.shape

  print data_train.columns
  target_train = data_train['target']
  print target_train
  data_train.drop('target',axis=1,inplace=True)
  target_test = data_test['target']
  start = time.clock()
  data_test.drop('target',axis=1,inplace=True)

  rfr = RandomForestRegressor(n_jobs=-1)
  rfr = rfr.fit(data_train,target_train)
  predict_loc_regres = rfr.predict(data_test)
  score = rfr.score(nz_data_test,nz_target_test)
  end = time.clock()

  rf_times += [end-start]
  predicts_class += [predict_loc_class]
  predicts += [predict]
  scores +=[score]
    ##training_scores +=[training_score]
  gn = normalized_weighted_gini(target_test,predict_loc_class,data_test.var11)
  #gn = normalized_weighted_gini(target_test,predict,data_test.var11)
  ginis +=[gn]

print "rf_times",np.array(rf_times).mean(),"x",len(rf_times)
#print "training_scores: ",training_scores, pd.DataFrame(training_scores).mean()
print "correct_targets = " , np.array(correct_targets).mean(), np.array(correct_targets).std(), np.array(correct_targets).ptp()
print "score: ", np.array(scores).mean(), np.array(scores).std(), np.array(scores).ptp()
print "gini: ", np.array(ginis).mean(), np.array(ginis).std(), np.array(ginis).ptp()

#outputfile_name = open(os.path.splitext(sys.argv[0])[0]+'_prefix.txt','wt')

#print >> outputfile_name, outputfile

#print outputfile + "_f" + str(folds)
