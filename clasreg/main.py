#!/bin/python
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
#from sklearn.metrics import accuracy_score
from bootstrap_znz import bootstrap_znz
from gini import normalized_weighted_gini

def get_nz(target_train,data_train):
    zeros = []
    nonzeros = []
    for index, row in target_train.iteritems():
        if row:
            nonzeros += [index]
        else:
            zeros += [index]
    nz_target_train = target_train.drop(zeros)
    nz_data_train = data_train.drop(zeros)
    return nz_target_train,nz_data_train

folds=10
rf_times = []
scores = []
training_scores = []
predicts = []
correct_targets = []
ginis = []
for i in range(folds):
    filename = ("../prep/oh_imp_train_%d.csv" % (i))
    data_train = pd.read_csv(filename)
    target_train = data_train['target']
    data_train.drop('target',axis=1,inplace=True)
    filename = ("../prep/oh_imp_test_%d.csv" % (i))
    data_test = pd.read_csv(filename)
    target_test = data_test['target']
    data_test.drop('target',axis=1,inplace=True)

    nz_target_train,nz_data_train = get_nz(target_train,data_train)

    target_train[target_train!=0]=1
    data_train.drop('target',axis=1,inplace=True)
    target_test[target_test!=0]=1
    data_test.drop('target',axis=1,inplace=True)

    target_train,data_train = bootstrap_znz(target_train,data_train)

    start = time.clock()
    rfc= RandomForestClassifier(n_jobs=-1)
    rfr = RandomForestRegressor(n_estimators = 100,n_jobs=-1)
    rfc = rfc.fit(data_train,target_train)
    end = time.clock()

    predict_loc = rfc.predict(data_test)
    correct_targets += [float(len(target_test[target_test==1])-len(predict_loc[predict_loc==1]))/len(predict_loc)]
    ##training_score = rfc.score(data_train,target_train)
    ##score = rfc.score(data_test,target_test)

    rf_times += [end-start]
    predicts += [predict_loc]
    ##scores +=[score]
    ##training_scores +=[training_score]
    gn = normalized_weighted_gini(target_test,predict_loc,data_test.var11)
    ginis +=[gn]

print "rf_times",np.array(rf_times).mean(),"x",len(rf_times)
#print "training_scores: ",training_scores, pd.DataFrame(training_scores).mean()
print "correct_targets = " , np.array(correct_targets).mean(), correct_targets
print "gini: ", np.array(ginis).mean(), np.array(ginis)
#print "score: ", np.array(scores).mean()
