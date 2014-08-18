import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.decomposition import PCA
import numpy as np
import pylab
import time

data = pd.read_csv('train.csv',nrows=10000)
imp_nan = Imputer(missing_values=np.nan, strategy='mean', axis=0)
# print data[0:0]
samples = data.ix[:,'crimeVar1':'weatherVar236']
print samples.shape
samples = imp_nan.fit_transform(samples)
labels = data.ix[:,'target']
nonzero =0
test_id = []
for i in range(0,len(labels)): # get nonzero labels
    if labels[i]:
        print i,"  ",labels[i]
        test_id+= [i]
        nonzero+=1
print nonzero
test = data.ix[test_id,'crimeVar1':'weatherVar236']
test = imp_nan.fit_transform(test)
pca = PCA(n_components=282)
pca.fit(samples)
samples = pca.transform(samples)
test = pca.transform(test)

forest = RandomForestClassifier(n_estimators = 100)
start = time.clock()
forest = forest.fit(samples,labels)
end = time.clock()
print "time to train = ", end-start
output = forest.predict(test)
print sum(x!=0 for x in output)
print output
print forest.predict_proba(test)

