#!/bin/python

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

data  = pd.read_csv('train.csv',nrows=10000)

def col_to_list(d,v):
    dv= d.ix[:,v]
    enum = list(set(dv))
    enum.insert(len(enum),enum.pop(enum.index('Z'))) #move Z (missing value) to be the last element
    for x in range(0,len(dv)):
        for i in range(0,len(enum)):
            if dv[x] == enum[i]:
                dv[x] = i;
    return dv


def data_from_var(data,vars):
    d =[]
    for i in vars:
        d.append(col_to_list(data,i))
    return d

d = data_from_var(data,['var1','var2','var3','var4'])

enc = OneHotEncoder()
enc.fit(zip(*d))
print enc.active_features_
print enc.feature_indices_
print enc.n_values_
td = enc.transform(zip(*d))
print td.shape
print td[0:10].toarray()
