import pandas as pd
import onehot
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

data = pd.read_csv('train.csv',nrows=10000)
print data.shape
categorical_samples = onehot.transform(data,['var1','var2','var3','var4','var5','var6','var7','var8','var9','dummy'])
categorical_samples = pd.DataFrame(categorical_samples.toarray())
continuous_samples = data.ix[:,'var10':'weatherVar236']
samples = pd.concat([categorical_samples,continuous_samples],axis=1)
imp_nan = Imputer(missing_values=np.nan, strategy='mean', axis=0)
imp_nan.fit(samples)
samples = imp_nan.transform(samples)
print samples.shape
#print samples

labels = data.ix[:,'target']
nonzero =0
test_id = []
for i in range(0,len(labels)): # get nonzero labels
    if labels[i]:
        print i,"  ",labels[i]
        test_id+= [i]
        nonzero+=1
categorical_test = categorical_samples.ix[test_id,:]
continuous_test = continuous_samples.ix[test_id,:]
test = pd.concat([categorical_test,continuous_test],axis=1)
test = imp_nan.transform(test)

pca = PCA(n_components=50)
pca.fit(samples)
samples = pca.transform(samples)
test = pca.transform(test)

forest = RandomForestClassifier(n_estimators = 100,n_jobs=-1)
start = time.clock()
forest = forest.fit(samples,labels)
end = time.clock()
print "time to train = ", end-start
output = forest.predict(test)
print sum(x!=0 for x in output)
print output
print forest.predict_proba(test)


