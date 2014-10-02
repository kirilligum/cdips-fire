#!/bin/python
from __future__ import division
import pandas as pd
#import numpy as np

def weighted_gini(act,pred,weight):
    df = pd.DataFrame({"act":act,"pred":pred,"weight":weight})
    df = df.sort('pred',ascending=False)
    df["random"] = (df.weight / df.weight.sum()).cumsum()
    total_pos = (df.act * df.weight).sum()
    df["cum_pos_found"] = (df.act * df.weight).cumsum()
    df["lorentz"] = df.cum_pos_found / total_pos
    #n = df.shape[0]
    #df["gini"] = (df.lorentz - df.random) * df.weight
    #return df.gini.sum()
    gini = sum(df.lorentz[1:].values * (df.random[:-1])) - sum(df.lorentz[:-1].values * (df.random[1:]))
    return gini

def normalized_weighted_gini(act,pred,weight):
    return weighted_gini(act,pred,weight) / weighted_gini(act,act,weight)

##### Test:
#test_var11 = pd.Series([1,2,5,4,3])
#test_pred = pd.Series([0.1, 0.4, 0.3, 1.2, 0.0])
#test_target = pd.Series([0, 0, 1, 0, 1])
#print "weighted_gini", weighted_gini(test_target,test_pred,test_var11)
#print "normalized_weighted_gini", normalized_weighted_gini(test_target,test_pred,test_var11)

#var11 <- c(1, 2, 5, 4, 3)
#pred <- c(0.1, 0.4, 0.3, 1.2, 0.0)
#target <- c(0, 0, 1, 0, 1)

#should now score -0.821428571428572.
