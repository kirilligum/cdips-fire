import pandas as pd
from onehot import onehot
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.decomposition import PCA
import time
import random
#import pickle

#data = pd.read_csv('train.csv',nrows=1000)
#data = pd.read_csv('train.csv',nrows=100000)
print " starting data load"
start = time.clock()
data = pd.read_csv('../znz_data/znz_subset.csv')
print data.shape
end = time.clock()
print "time = ", end-start


print " starting onehot"
start = time.clock()
oh = onehot()
oh.fit(data,['var1','var2','var3','var4','var5','var6','var7','var8','var9','dummy'])
categorical_samples = oh.transform(data)
continuous_var_samples = data.ix[:,'var10':'var17']
continuous_other_samples = data.ix[:,'crimeVar1':'weatherVar236']
samples = pd.concat([categorical_samples,continuous_var_samples,continuous_other_samples],axis=1)
end = time.clock()
print "time = ", end-start

print " starting impute"
start = time.clock()
imp_nan = Imputer(missing_values=np.nan, strategy='mean', axis=0)
imp_nan.fit(samples)
samples = imp_nan.transform(samples)
print samples.shape
end = time.clock()
print "time = ", end-start

print " starting pca"
start = time.clock()
pca = PCA(n_components=50)
pca.fit(samples)
samples = pca.transform(samples)
end = time.clock()
print "time = ", end-start

print " starting to dataframe"
start = time.clock()
print samples.shape
samples_df = pd.DataFrame(samples)
prep = pd.concat([samples_df,data.ix[:,'target']],axis=1)
end = time.clock()
print "time = ", end-start

print " starting shuffle rows"
start = time.clock()
rows = list(prep.index)
random.shuffle(rows)
prep = prep.ix[rows]
end = time.clock()
print "time = ", end-start

print " starting to csv"
start = time.clock()
prep.to_csv('prep.csv')
end = time.clock()
print "time = ", end-start

