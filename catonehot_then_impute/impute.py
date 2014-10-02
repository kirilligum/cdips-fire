
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
import time
import sys
import os
#import random

log = open(os.path.splitext(sys.argv[0])[0]+'.log','wt')

data_train = pd.read_csv(sys.argv[1])
data_test = pd.read_csv(sys.argv[2])
#data = pd.concat([data_train,data_test])
#print >> log,  data_train.shape, "  ", data_test.shape ,"  " , data.shape

data_train.replace('Z','NaN', inplace = True)
data_test.replace('Z','NaN', inplace = True)
print data_train

imp_median = Imputer(missing_values=np.nan, strategy='median', axis=0)
imp_median.fit(pd.concat([data_train.drop('target',1),data_test]))
train_imp = imp_nan.transform(data_train)
test_imp = imp_nan.transform(data_test)
print train_imp


log.close()
