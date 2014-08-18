import pandas as pd
from onehot import onehot
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.decomposition import PCA
#from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
#import selector
#import preprocess
#reload(preprocess)
#import gini
import time
#import pickle

#data = pd.read_csv('train.csv',nrows=1000)
#data = pd.read_csv('train.csv',nrows=100000)
print " starting data load"
start = time.clock()
data = pd.read_csv('znz_data/znz_subset.csv')
print data.shape
end = time.clock()
print "time = ", end-start


print " starting onehot"
start = time.clock()
### onehot
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


print " starting test set"
start = time.clock()
labels = data.ix[:,'target']
nonzero =0
test_id = []
for i in range(0,len(labels)): # get nonzero labels
    if labels[i]:
        #print i,"  ",labels[i]
        test_id+= [i]
        nonzero+=1
categorical_test = categorical_samples.ix[test_id,:]
continuous_var_test = continuous_var_samples.ix[test_id,:]
continuous_other_test = continuous_other_samples.ix[test_id,:]
test = pd.concat([categorical_test,continuous_var_test,continuous_other_test],axis=1)
test = imp_nan.transform(test)
end = time.clock()
print "time = ", end-start

print " starting pca"
start = time.clock()
pca = PCA(n_components=50)
pca.fit(samples)
samples = pca.transform(samples)
test = pca.transform(test)
end = time.clock()
print "time = ", end-start

print" starting rf"
start = time.clock()
forest = RandomForestClassifier(n_estimators = 100,n_jobs=-1)
forest = forest.fit(samples,labels)
end = time.clock()
print "time to train = ", end-start
output = forest.predict(test)
print sum(x!=0 for x in output)
print output
print forest.predict_proba(test)


