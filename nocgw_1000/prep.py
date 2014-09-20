import pandas as pd
from onehot import onehot
import numpy as np
from sklearn.preprocessing import Imputer
#from sklearn.decomposition import PCA
import time
import sys
import os
#import random
#import pickle

log = open(os.path.splitext(sys.argv[0])[0]+'.log','wt')

#filename = ("../data/subtrain.csv")
#filename = ("znz_train_0.csv")
#filename = ("../data/train.csv")
#data_train = pd.read_csv(filename)
data_train = pd.read_csv(sys.argv[1])
#filename = ("../data/subtest.csv")
#filename = ("../data/test.csv")
#data_test = pd.read_csv(filename)
data_test = pd.read_csv(sys.argv[2])
data = pd.concat([data_train,data_test])
print >> log,  data_train.shape, "  ", data_test.shape ,"  " , data.shape


print >> log,  " starting onehot"
start = time.clock()
oh = onehot()
oh.fit(data,['var1','var2','var3','var4','var5','var6','var7','var8','var9','dummy'])
categorical_train = oh.transform(data_train)
continuous_var_train = data_train.ix[:,'var10':'var17']
#continuous_other_train = data_train.ix[:,'crimeVar1':'weatherVar236']
id_train = data_train.ix[:,'id']
#train  = pd.concat([id_train,categorical_train,continuous_var_train,continuous_other_train],axis=1)
train  = pd.concat([id_train,categorical_train,continuous_var_train],axis=1)
categorical_test = oh.transform(data_test)
continuous_var_test = data_test.ix[:,'var10':'var17']
#continuous_other_test = data_test.ix[:,'crimeVar1':'weatherVar236']
id_test = data_test.ix[:,'id']
#test  = pd.concat([id_test,categorical_test,continuous_var_test,continuous_other_test],axis=1)
test  = pd.concat([id_test,categorical_test,continuous_var_test],axis=1)
end = time.clock()
total = pd.concat([train,test])
print >> log,  train.shape, "  ", test.shape ,"  " , total.shape
print >> log,  "time = ", end-start
print >> log,  id_train
print >> log,  id_test

print >> log,  " starting impute"
start = time.clock()
imp_nan = Imputer(missing_values=np.nan, strategy='mean', axis=0)
imp_nan.fit(total)
train_imp = imp_nan.transform(train)
test_imp = imp_nan.transform(test)
print >> log,  train_imp.shape, "  ", test_imp.shape ,"  " , total.shape
end = time.clock()
print >> log,  "time = ", end-start

#filename = ("subtrain.csv")
train_filename = (os.path.splitext(os.path.basename(sys.argv[1]))[0]+"prep.csv")
train_columns = train.columns
train = pd.DataFrame(train_imp, columns=train_columns)
train = pd.concat([train,data_train.ix[:,'target']],axis=1)
train.to_csv(train_filename,index=0)
#filename = ("subtest.csv")
#filename = ("test.csv")
test_filename = (os.path.splitext(os.path.basename(sys.argv[2]))[0]+"prep.csv")
test_columns = test.columns
test = pd.DataFrame(test_imp, columns=test_columns)
if 'target' in data_test:
    test = pd.concat([test,data_test.ix[:,'target']],axis=1)
test.to_csv(test_filename,index=0)

log.close()

print train_filename, test_filename
