#!/bin/python
import pandas as pd
import random
import sys
import os

def bootstrap_znz(target_train,data_train,znz_ratio=1):
    zeros = []
    nonzeros = []
    for index, row in target_train.iteritems():
        if row:
            nonzeros += [index]
        else:
            zeros += [index]
    #print "nonzeros = ",len(nonzeros), "; zeros = ", len(zeros)
    bootstrap_target_train = pd.Series()
    bootstrap_data_train = pd.DataFrame(columns= data_train.columns)
    n_bootstrap_samples = len(nonzeros)+int((len(zeros)-len(nonzeros))*znz_ratio)
    samples = []
    for i in range(n_bootstrap_samples):
        samples += [random.choice(nonzeros)]
    bootstrap_target_train = bootstrap_target_train.append(target_train[samples])
    bootstrap_data_train = bootstrap_data_train.append(data_train.ix[samples,:])
    target_train.drop(nonzeros,inplace=1)
    data_train.drop(nonzeros,inplace=1)
    target_train = target_train.append(bootstrap_target_train)
    data_train = data_train.append(bootstrap_data_train)
    return target_train,data_train

log = open(os.path.splitext(sys.argv[0])[0]+'.log','wt')

data_train = pd.read_csv(sys.argv[1])
print >> log,  data_train.shape
target_train = data_train['target']
data_train.drop('target',axis=1,inplace=True)

target_train,data_train = bootstrap_znz(target_train,data_train)
out_filename = (os.path.splitext(os.path.basename(sys.argv[1]))[0]+"_znz.csv")
train = pd.concat([target_train,data_train],axis=1)
train.to_csv(out_filename,index=0)
