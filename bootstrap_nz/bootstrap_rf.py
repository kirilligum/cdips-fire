#!/bin/python
import pandas as pd
import numpy as np
import time
#from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
#from sklearn.metrics import accuracy_score
from bootstrap_znz import bootstrap_znz
from gini import normalized_weighted_gini

folds=10
rf_times = []
scores = []
training_scores = []
predicts = []
ginis = []
for i in range(10):
    filename = ("../prep/oh_imp_train_%d.csv" % (i))
    data_train = pd.read_csv(filename)
    target_train = data_train['target']
    data_train.drop('target',axis=1,inplace=True)
    filename = ("../prep/oh_imp_test_%d.csv" % (i))
    data_test = pd.read_csv(filename)
    target_test = data_test['target']
    data_test.drop('target',axis=1,inplace=True)

    target_train,data_train = bootstrap_znz(target_train,data_train)

    #print" starting rf"
    start = time.clock()
    forest = RandomForestRegressor(n_estimators = 100,n_jobs=-1)
    #forest = RandomForestRegressor(n_jobs=-1)
    forest = forest.fit(data_train,target_train)
    end = time.clock()
    rf_times += [end-start]
    predict_loc = forest.predict(data_test)
    predicts += [predict_loc]
    #print sum(x!=0 for x in output)
    training_score = forest.score(data_train,target_train)
    score = forest.score(data_test,target_test)
    scores +=[score]
    training_scores +=[training_score]
    gn = normalized_weighted_gini(target_test,predict_loc,data_test.var11)
    ginis +=[gn]

print "rf_times",np.array(rf_times).mean()
#print "training_scores: ",training_scores, pd.DataFrame(training_scores).mean()
print "gini: ", np.array(ginis).mean(),
print "score: ", np.array(scores).mean()

