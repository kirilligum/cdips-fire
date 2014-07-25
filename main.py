#!/bin/python

import pandas as pd
import onehot

data  = pd.read_csv('train.csv',nrows=10000)

td = onehot.transform(data,['var1','var2','var3','var4'])

print td.shape
print td[0:10].toarray()
