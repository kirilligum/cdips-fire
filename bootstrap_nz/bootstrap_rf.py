#!/bin/python
import pandas as pd
#import numpy as np
import time
#from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
#from sklearn.metrics import accuracy_score
import random
from gini import normalized_weighted_gini

folds=10
znz_ratio = 1
scores = []
training_scores = []
predicts = []
ginis = []
for i in range(1):
    filename = ("../prep/oh_imp_train_%d.csv" % (i))
    data_train = pd.read_csv(filename)
    target_train = data_train['target']
    data_train.drop('target',axis=1,inplace=True)
    filename = ("../prep/oh_imp_test_%d.csv" % (i))
    data_test = pd.read_csv(filename)
    target_test = data_test['target']
    data_test.drop('target',axis=1,inplace=True)

    zeros = []
    nonzeros = []


    for index, row in target_train.iteritems():
        if row:
            nonzeros += [index]
        else:
            zeros += [index]

    print len(nonzeros)
    print len(zeros)

    bootstrap_target_train = target_train[[random.choice(nonzeros)]]
    bootstrap_target_train = bootstrap_target_train.append(target_train[[random.choice(nonzeros)]])
    bootstrap_data_train = data_train.ix[random.choice(nonzeros),:]
    print bootstrap_target_train.shape, type(bootstrap_target_train),bootstrap_data_train.shape

    #n_bootstrap_samples = len(nonzeros)
    #n_bootstrap_samples = len(nonzeros)+int((len(zeros)-len(nonzeros))*znz_ratio)
    #samples = []
    #for i in range(n_bootstrap_samples-1):
        #samples += [random.choice(nonzeros)]
    #print samples
    #bootstrap_target_train = bootstrap_target_train.append(target_train[samples])
    #print bootstrap_data_train.shape, data_train.shape, bootstrap_target_train.shape
    #bootstrap_data_train = pd.concat([bootstrap_data_train,data_train[samples]])
    #bootstrap_data_train = bootstrap_data_train.append(data_train[samples])

    #print "n_bootstrap_samples = ",n_bootstrap_samples
    print "bootstrap_target_train",len(bootstrap_target_train),bootstrap_target_train.values
    #print "target_train",target_train[nonzeros]
    #print "bootstrap_data_train",len(bootstrap_data_train),bootstrap_data_train.values
    #print "data_train",data_train[nonzeros]

    #print "",data_train.shape, "  ", data_test.shape ,"  " , target_train.shape, " ", target_test.shape

    #print" starting rf"
    start = time.clock()
    forest = RandomForestRegressor(n_estimators = 100,n_jobs=-1)
    #forest = RandomForestRegressor(n_jobs=-1)
    forest = forest.fit(data_train,target_train)
    end = time.clock()
    print "time to train = ", end-start
    predict_loc = forest.predict(data_test)
    predicts += [predict_loc]
    #print sum(x!=0 for x in output)
    training_score = forest.score(data_train,target_train)
    score = forest.score(data_test,target_test)
    scores +=[score]
    training_scores +=[training_score]
    gn = normalized_weighted_gini(target_test,predict_loc,data_test.var11)
    ginis +=[gn]

    #df= pd.concat([pd.DataFrame(predict_loc,columns=['predicted']),target_test,data_test['var11']],axis=1)
    #df.sort(columns='predicted')
    #df["base"] = (df['var11'] / df['var11'].sum()).cumsum()
    #print df
    #total_pos = (df.target * df['var11']).sum()
    #df["cum_pos_found"] = (df.target * df.var11).cumsum()
    #df["lorentz"] = df.cum_pos_found / total_pos
    #df["gini"] = (df.lorentz - df.base) * df['var11']
    #print df.gini.sum()

#print "gini: ",ginis, pd.DataFrame(ginis).describe()
#print "training_scores: ",training_scores, pd.DataFrame(training_scores).describe()
#print "score: ",scores, pd.DataFrame(scores).describe()

