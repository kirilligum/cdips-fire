import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
#from sklearn.decomposition import PCA
import time
import sys
import os
import itertools
#import random
#import pickle

def nominal_binary(dtin,d):
  dt = dtin.replace({'Z':np.nan})
  d.replace({'Z':np.nan},inplace=1)
  enum = list(set(dt.dropna()))
  m = dict(zip(enum,range(len(enum))))
  return d.replace(m,inplace=1)

def nominal(dtin,d):
  dt = dtin.replace({'Z':np.nan})
  d.replace({'Z':np.nan},inplace=1)
  enum = list(set(dt.dropna()))
  m = dict(zip(enum,range(len(enum))))
  dt.replace(m,inplace=1)
  d.replace(m,inplace=1)
  enc =  OneHotEncoder().fit([[x] for x in dt.dropna().values.tolist()])
  nans = [1 if np.isnan(x).any() else 0 for x in d ]
  d.fillna(d.dropna().median(),inplace=1)
  td = enc.transform([[x] for x in d.values.tolist()]).todense()
  for j,v in enumerate(td):
    if nans[j]:
      td[j]=np.nan
  return pd.DataFrame(td,columns = [dt.name+"_"+str(x) for x in range(td.shape[1])])

def transform_var4(dtin,d):
  nans = [1 if x=='Z' else 0 for x in d ]
  dt = dtin.replace({'Z':d[0]})
  d.replace({'Z':d[0]},inplace=1)
  letters = set([x[0] for x in set(dt)])
  td = pd.DataFrame(index=d.index,columns=[d.name+'_'+l for l in letters])
  for i,row in d.iteritems():
    if not nans[i]:
      td.iloc[i]=0
      td.iloc[i][d.name+'_'+d[i][0]]=d[i][1:]
  return td

log = open(os.path.splitext(sys.argv[0])[0]+'.log','wt')

data_train = pd.read_csv(sys.argv[1])
data_train_nt = data_train.drop('target',axis=1)
data_test = pd.read_csv(sys.argv[2])
data_test_nt = data_test
if 'target' in data_test:
  data_test_nt.drop('target',axis=1,inplace=True)
#data = pd.concat([data_train,data_test])
print >> log,  data_train.shape, "  ", data_test.shape
#print >> log,  data_train.shape, "  ", data_test.shape ,"  " , data.shape

continuous_other_train = data_train_nt.ix[:,'crimeVar1':'weatherVar236']
data_train_nt.drop(continuous_other_train,axis=1,inplace=1)

print >> log,  " starting onehot"
start = time.clock()

datafit = pd.concat([data_train_nt,data_test_nt])

for i in ['dummy','var9']:
  nominal_binary(datafit[i],data_train[i])
  nominal_binary(datafit[i],data_test[i])
for i in ['var2','var5','var6']:
  data_train = pd.concat([data_train,nominal(datafit[i],data_train[i])],axis=1).drop(i,axis=1)
  data_test = pd.concat([data_test,nominal(datafit[i],data_test[i])],axis=1).drop(i,axis=1)
data_train = pd.concat([data_train,transform_var4(datafit['var4'],data_train['var4'])],axis=1).drop('var4',axis=1)
data_test = pd.concat([data_test,transform_var4(datafit['var4'],data_test['var4'])],axis=1).drop('var4',axis=1)
for i in ['var1','var3','var7','var8']:
  data_train[i].replace({'Z':np.nan},inplace=1)
  data_test[i].replace({'Z':np.nan},inplace=1)
print >>log,data_train.shape
print >>log,data_test.shape

end = time.clock()
print >> log,  "time = ", end-start

train_filename = (os.path.splitext(os.path.basename(sys.argv[1]))[0]+"_oh.csv")
data_train.to_csv(train_filename,index=0)
test_filename = (os.path.splitext(os.path.basename(sys.argv[2]))[0]+"_oh.csv")
data_test.to_csv(test_filename,index=0)

log.close()

print train_filename, test_filename
