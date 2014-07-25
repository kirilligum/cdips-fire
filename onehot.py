#!/bin/python

from sklearn.preprocessing import OneHotEncoder

def col_to_list(d,v):
    dv= d.ix[:,v]
    enum = list(set(dv))
    enum.insert(len(enum),enum.pop(enum.index('Z'))) #move Z (missing value) to be the last element
    for x in range(0,len(dv)):
        for i in range(0,len(enum)):
            if dv[x] == enum[i]:
                dv[x] = i;
    return dv


def transform(data,vars):
    d =[]
    for i in vars:
        d.append(col_to_list(data,i))
    enc = OneHotEncoder()
    enc.fit(zip(*d))
    #print enc.active_features_
    #print enc.feature_indices_
    #print enc.n_values_
    td = enc.transform(zip(*d))
    return td

