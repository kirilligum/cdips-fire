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

data_train = pd.read_csv(sys.argv[1])
data_test = pd.read_csv(sys.argv[2])
data = pd.concat([data_train,data_test])
print >> log,  data_train.shape, "  ", data_test.shape ,"  " , data.shape


print >> log,  " starting onehot"
start = time.clock()
oh = onehot()
#oh.fit(data,['var1','var2','var3','var4','var5','var6','var7','var8','var9','dummy'])
#oh.fit(data,['var5','dummy'])
oh.fit(data,['var2','var4','var5','var6','var9','dummy'])
categorical_train = oh.transform(data_train)
#print "done with onehot"
#print categorical_train.columns
cator_col = ['var1','var3','var7','var8']
continuous_var_train = pd.concat([data_train.ix[:,'var10':'var17'],data_train[cator_col]], axis=1)
#continuous_other_train = data_train.ix[:,'crimeVar1':'weatherVar236']
id_train = data_train.ix[:,'id']
#train  = pd.concat([id_train,categorical_train,continuous_var_train,continuous_other_train],axis=1)
train  = pd.concat([id_train,categorical_train,continuous_var_train],axis=1)
#print sorted(list(train.columns))
categorical_test = oh.transform(data_test)
continuous_var_test = pd.concat([data_test.ix[:,'var10':'var17'],data_test[cator_col]], axis=1)
#continuous_other_test = data_test.ix[:,'crimeVar1':'weatherVar236']
id_test = data_test.ix[:,'id']
#test  = pd.concat([id_test,categorical_test,continuous_var_test,continuous_other_test],axis=1)
test  = pd.concat([id_test,categorical_test,continuous_var_test],axis=1)
#print sorted(list(test.columns))
train.replace(to_replace='Z', value=np.nan,inplace=True)
test.replace(to_replace='Z', value=np.nan,inplace=True)
end = time.clock()
print >> log,  "time = ", end-start
print >> log,  id_train
print >> log,  id_test

train_filename = (os.path.splitext(os.path.basename(sys.argv[1]))[0]+"_ohtrans.csv")
train = pd.concat([train,data_train.ix[:,'target']],axis=1)
train.to_csv(train_filename,index=0)
test_filename = (os.path.splitext(os.path.basename(sys.argv[2]))[0]+"_ohtrans.csv")
if 'target' in data_test:
    test = pd.concat([test,data_test.ix[:,'target']],axis=1)
test.to_csv(test_filename,index=0)

log.close()

print train_filename, test_filename
