import pandas as pd
from onehot import onehot
import numpy as np
from sklearn.preprocessing import Imputer
#from sklearn.decomposition import PCA
import time
#import random
#import pickle

folds=10
for i in range(folds):
    #filename = ("../znz_subset/znz_train_%d.csv" % (i))
    filename = ("../znz_data/znz_train_%d.csv" % (i))
    data_train = pd.read_csv(filename)
    #filename = ("../znz_subset/znz_test_%d.csv" % (i))
    filename = ("../znz_data/znz_test_%d.csv" % (i))
    data_test = pd.read_csv(filename)
    data = pd.concat([data_train,data_test])
    print data_train.shape, "  ", data_test.shape ,"  " , data.shape


    print " starting onehot"
    start = time.clock()
    oh = onehot()
    oh.fit(data,['var1','var2','var3','var4','var5','var6','var7','var8','var9','dummy'])
    categorical_train = oh.transform(data_train)
    continuous_var_train = data_train.ix[:,'var10':'var17']
    continuous_other_train = data_train.ix[:,'crimeVar1':'weatherVar236']
    train  = pd.concat([categorical_train,continuous_var_train,continuous_other_train],axis=1)
    categorical_test = oh.transform(data_test)
    continuous_var_test = data_test.ix[:,'var10':'var17']
    continuous_other_test = data_test.ix[:,'crimeVar1':'weatherVar236']
    test  = pd.concat([categorical_test,continuous_var_test,continuous_other_test],axis=1)
    end = time.clock()
    total = pd.concat([train,test])
    print train.shape, "  ", test.shape ,"  " , total.shape
    print "time = ", end-start

    print " starting impute"
    start = time.clock()
    imp_nan = Imputer(missing_values=np.nan, strategy='mean', axis=0)
    imp_nan.fit(total)
    train_imp = imp_nan.transform(train)
    test_imp = imp_nan.transform(test)
    print train_imp.shape, "  ", test_imp.shape ,"  " , total.shape
    end = time.clock()
    print "time = ", end-start

    filename = ("oh_imp_all_train_%d.csv" % (i))
    train_columns = train.columns
    train = pd.DataFrame(train_imp, columns=train_columns)
    outdata = pd.concat([train,data_train.ix[:,'target']],axis=1)
    outdata.to_csv(filename)
    filename = ("oh_imp_all_test_%d.csv" % (i))
    test_columns = test.columns
    test = pd.DataFrame(test_imp, columns=test_columns)
    outdata = pd.concat([test,data_test.ix[:,'target']],axis=1)
    outdata.to_csv(filename)

#print " starting data load"
#start = time.clock()
#data = pd.read_csv('../znz_data/znz_subset.csv')
#print data.shape
#end = time.clock()
#print "time = ", end-start

#print " starting onehot"
#start = time.clock()
#oh = onehot()
#oh.fit(data,['var1','var2','var3','var4','var5','var6','var7','var8','var9','dummy'])
#categorical_samples = oh.transform(data)
#continuous_var_samples = data.ix[:,'var10':'var17']
#continuous_other_samples = data.ix[:,'crimeVar1':'weatherVar236']
#samples = pd.concat([categorical_samples,continuous_var_samples,continuous_other_samples],axis=1)
#end = time.clock()
#print "time = ", end-start

#print " starting impute"
#start = time.clock()
#imp_nan = Imputer(missing_values=np.nan, strategy='mean', axis=0)
#imp_nan.fit(samples)
#samples = imp_nan.transform(samples)
#print samples.shape
#end = time.clock()
#print "time = ", end-start

#print " starting pca"
#start = time.clock()
#pca = PCA(n_components=50)
#pca.fit(samples)
#samples = pca.transform(samples)
#end = time.clock()
#print "time = ", end-start

#print " starting to dataframe"
#start = time.clock()
#print samples.shape
#samples_df = pd.DataFrame(samples)
#prep = pd.concat([samples_df,data.ix[:,'target']],axis=1)
#end = time.clock()
#print "time = ", end-start

#print " starting shuffle rows"
#start = time.clock()
#rows = list(prep.index)
#random.shuffle(rows)
#prep = prep.ix[rows]
#end = time.clock()
#print "time = ", end-start

#print " starting to csv"
#start = time.clock()
#prep.to_csv('prep.csv')
#end = time.clock()
#print "time = ", end-start

