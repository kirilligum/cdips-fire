#!/bin/python
import pandas as pd
#import numpy as np
import time
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import cross_validation
#from sklearn.metrics import accuracy_score
#from sklearn.metrics import r2_score
from gini import normalized_weighted_gini
#import random
import sys
import os

class mixmodels:
    def __init__(self,nest=10):
        self.nest = nest
    def fit(self,data_train,target):
        self.target_train = target
        self.catcol = data_train.filter(like='var').columns.tolist()
        #start_gbr_tr = time.clock()
        self.gbr = GradientBoostingRegressor(n_estimators =self.nest,max_depth=7)
        self.gbr.fit(data_train,self.target_train)
        self.transformed_train_gbr = self.gbr.transform(data_train,threshold="0.35*mean")
        self.gbr_tr_fit = GradientBoostingRegressor(n_estimators =self.nest,max_depth=7)
        self.gbr_tr_fit.fit(self.transformed_train_gbr,self.target_train)
        #end_gbr_tr = time.clock()
        #print >> log, "time_gbr_tr = ", end_gbr_tr-start_gbr_tr

        #start_xfr_tr = time.clock()
        self.xfr= ExtraTreesRegressor(n_estimators =self.nest,max_depth=7)
        self.xfr.fit(data_train,self.target_train)
        self.transformed_train_xfr = self.xfr.transform(data_train,threshold="0.35*mean")
        self.xfr_tr_fit = ExtraTreesRegressor(n_estimators =self.nest,max_depth=7)
        self.xfr_tr_fit.fit(self.transformed_train_xfr,self.target_train)
        #end_xfr_tr = time.clock()
        #print >> log, "time_xfr_tr = ", end_xfr_tr-start_xfr_tr

        #start_gbr_cat = time.clock()
        self.gbr_cat_fit = GradientBoostingRegressor(n_estimators =self.nest,max_depth=7)
        self.gbr_cat_fit.fit(data_train[self.catcol],self.target_train)
        #end_gbr_cat = time.clock()
        #print >> log, "time_gbr_cat = ", end_gbr_cat-start_gbr_cat

        #start_xfr_cat = time.clock()
        self.xfr_cat_fit = ExtraTreesRegressor(n_estimators =self.nest,max_depth=7)
        self.xfr_cat_fit.fit(data_train[self.catcol],self.target_train)
        #end_xfr_cat = time.clock()
        #print >> log, "time_xfr_cat = ", end_xfr_cat-start_xfr_cat
        return self

    def predict(self,data_test):
        mix_test_list = []

        transformed_test_gbr = self.gbr.transform(data_test,threshold="0.35*mean")
        mix_test_list += [pd.Series(self.gbr_tr_fit.predict(transformed_test_gbr))]

        transformed_test_xfr = self.xfr.transform(data_test,threshold="0.35*mean")
        mix_test_list += [pd.Series(self.xfr_tr_fit.predict(transformed_test_xfr))]

        mix_test_list += [pd.Series(self.gbr_cat_fit.predict(data_test[self.catcol]))]

        mix_test_list += [pd.Series(self.xfr_cat_fit.predict(data_test[self.catcol]))]

        mix_test = pd.concat(mix_test_list,1)

        mix_ave = mix_test.mean(1)
        mix_ave.name='target'

        return mix_ave
    def score(self,data_test,target_test):
        total_score = []
        transformed_test_gbr = self.gbr.transform(data_test,threshold="0.35*mean")
        total_score += [ self.gbr_tr_fit.score(transformed_test_gbr,target_test) ]
        transformed_test_xfr = self.xfr.transform(data_test,threshold="0.35*mean")
        total_score += [ self.xfr_tr_fit.score(transformed_test_xfr,target_test) ]
        total_score += [ self.gbr_cat_fit.score(data_test[self.catcol],target_test) ]
        total_score += [ self.xfr_cat_fit.score(data_test[self.catcol],target_test) ]
        return sum(total_score)/float(len(total_score))

    def gini(self,data_test,target_test):
        weight = data_test.var11
        gns = []
        transformed_test_gbr = self.gbr.transform(data_test,threshold="0.35*mean")
        gns += [normalized_weighted_gini(target_test.tolist(),self.gbr_tr_fit.predict(transformed_test_gbr).tolist(),weight.tolist()) ]
        transformed_test_xfr = self.xfr.transform(data_test,threshold="0.35*mean")
        gns += [normalized_weighted_gini(target_test.tolist(),self.xfr_tr_fit.predict(transformed_test_xfr).tolist(),weight.tolist()) ]
        gns += [normalized_weighted_gini(target_test.tolist(),self.gbr_cat_fit.predict(data_test[self.catcol]).tolist(),weight.tolist()) ]
        gns += [normalized_weighted_gini(target_test.tolist(),self.xfr_cat_fit.predict(data_test[self.catcol]).tolist(),weight.tolist()) ]
        return sum(gns)/float(len(gns))

log = open(os.path.splitext(sys.argv[0])[0]+'.log','wt')

data_train_in = pd.read_csv(sys.argv[1])
print >> log,  data_train_in.shape

start = time.clock()
target_train = data_train_in['target']
data_train = data_train_in.drop(['id','target'],axis=1)

scores = []
gns = []
skfclasses = [1 if x>0 else 0 for x in target_train]
skf = cross_validation.StratifiedKFold(skfclasses,n_folds=5)
for train_index, test_index in skf:
    mm = mixmodels(nest=10)
    mm.fit(data_train.iloc[train_index],target_train.iloc[train_index])
    scores += [mm.score(data_train.iloc[test_index],target_train.iloc[test_index])]
    gns += [mm.gini(data_train.iloc[test_index],target_train.iloc[test_index])]
    predict_loc_regres = mm.predict(data_train.iloc[test_index])
    #print 'gini inputs   ',target_train.iloc[test_index],predict_loc_regres,data_train.iloc[test_index].var11
    #gns += [normalized_weighted_gini(target_train.iloc[test_index].tolist(),predict_loc_regres.tolist(),data_train.iloc[test_index].var11.tolist() )]
score = sum(scores)/float(len(scores))
gn = sum(gns)/float(len(gns))
print scores
print gns
print'score = ', score,'   gn = ', gn

end = time.clock()


#out_filename = (os.path.splitext(os.path.basename(sys.argv[1]))[0])
#mix_test.to_csv(out_filename+"_mixmeth.csv")
#mix_ave.to_csv(out_filename+"_mixave.csv",header=1)
#print out_filename

log.close()
