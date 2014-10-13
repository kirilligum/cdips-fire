import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
import time
import sys
import os

log = open(os.path.splitext(sys.argv[0])[0]+'.log','wt')

data_train_in = pd.read_csv(sys.argv[1])
data_test_in = pd.read_csv(sys.argv[2])
train = data_train_in.drop('target',axis=1)
if 'target' in data_test_in:
  test = data_test_in.drop('target',axis=1)
else:
  test = data_test_in
data = pd.concat([train,test])
print >> log,  train.shape, "  ", test.shape ,"  " , data.shape

imp_discrete = Imputer(missing_values=np.nan, strategy='median', axis=0)
imp_continuous = Imputer(missing_values=np.nan, strategy='median', axis=0)
print >> log,sorted(list(data.columns))
original_nominal = ['var2','var4','var5','var6','var9','dummy']
discrete_nominal = [x for x in train.columns if any(y in x for y in original_nominal)]
#discrete_nominal = [x for x in train.columns if "_" in x]
discrete_ordinal = ['var1','var3','var7','var8']
discrete = discrete_nominal+discrete_ordinal
print >> log, "discrete", discrete
imp_discrete.fit(data[discrete])
continuous = train.ix[:,'var10':'var17'].columns
imp_continuous.fit(data[continuous])
train_imp_median = pd.DataFrame(imp_discrete.transform(train[discrete]),columns=train[discrete].columns)
train_imp_mean = pd.DataFrame(imp_continuous.transform(train[continuous]),columns=train[continuous].columns)
test_imp_median = pd.DataFrame(imp_discrete.transform(test[discrete]),columns=test[discrete].columns)
test_imp_mean = pd.DataFrame(imp_continuous.transform(test[continuous]),columns=test[continuous].columns)
id_train = data_train_in.ix[:,'id']
id_test = data_test_in.ix[:,'id']
train_imp = pd.concat([id_train,train_imp_median,train_imp_mean],axis=1)
test_imp = pd.concat([id_test,test_imp_median,test_imp_mean],axis=1)

train_filename = (os.path.splitext(os.path.basename(sys.argv[1]))[0]+"_imm.csv")
train_columns = train.columns
train = pd.DataFrame(train_imp, columns=train_columns)
train = pd.concat([train,data_train_in.ix[:,'target']],axis=1)
train.to_csv(train_filename,index=0)
test_filename = (os.path.splitext(os.path.basename(sys.argv[2]))[0]+"_imm.csv")
test_columns = test.columns
test = pd.DataFrame(test_imp, columns=test_columns)
if 'target' in data_test_in:
    test = pd.concat([test,data_test_in.ix[:,'target']],axis=1)
test.to_csv(test_filename,index=0)

print train_filename, test_filename

log.close()

