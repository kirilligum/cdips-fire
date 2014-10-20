import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn import mixture
import time
import sys
import os
import random
import itertools

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

log = open(os.path.splitext(sys.argv[0])[0]+'.log','wt')

data = pd.read_csv(sys.argv[1])
#data = pd.read_csv(sys.argv[1]).iloc[:10,[1]]
print >> log, "data.shape = " , data.shape

miss = data.apply(lambda x : x.apply(lambda y: np.isnan(y)))

imp_mean = Imputer(missing_values=np.nan, strategy='mean', axis=0)
data_imp_mean = pd.DataFrame(imp_mean.fit_transform(data),columns=data.columns)
print data_imp_mean.shape, data.shape
#print >> log, data_imp_mean, data

g = mixture.GMM(n_components=5, covariance_type='full')
g.fit(data_imp_mean)
print 'weights_ = ',g.weights_.shape
print >> log, 'weights_ = ',g.weights_.shape
print >> log, 'weights_ = ',g.weights_
#print 'means_ = ',g.means_.shape
#print >> log, 'means_ = ',g.means_.shape
#print >> log, 'means_ = ',g.means_
#print 'covars_ = ',g.covars_.shape
#print >> log, 'covars_ = ',g.covars_.shape
#print >> log, 'covars_ = ',g.covars_
#print 'converged_ = ',g.converged_
#print >> log, 'converged_ = ',g.converged_

### Decide from which gaussian to sample by sampling from weights
decide_node_sample = random.random()
print decide_node_sample
print g.weights_
iweight = 0
running_sum = 0
while decide_node_sample>running_sum:
  running_sum+=g.weights_[iweight]
  iweight+=1
  print "running_sum = ", running_sum
iweight-=1
print iweight

### separate the vector into known and unknown.

impute_sample = data.iloc[0]
nan_means = []
nan_covars = []
for i,imean, icovar in itertools.zip(impute_sample,g.means_,g.covars_):
  if np.isnan(i):
    nan_means += imean
    nan_covars += icovar
  else:
    prefactor= gaussian(imean,icov)
impran= np.random.multivariate_normal(nan_means,nan_cavs)




### calculate the coefficient from the known

## sample multivariate normal with uknown.

random_sample = np.random.multivariate_normal(g.means_[0],g.covars_[0])
print random_sample.shape

#tcols = {}
#nans ={}
#combine = []
#for i in self.cols:
  #d = self.col_to_list(data[i])
  #col_nans = []
  #col_nans_value = []
  #nonan = d.dropna()
  ##print "median", median
  #for j,v in enumerate(d):
    ##print j
    #if d[j] is np.nan:
      #col_nans += [1]
    #else:
      #col_nans += [0]

#train_filename = (os.path.splitext(os.path.basename(sys.argv[3]))[0]+"_imm.csv")
#train_columns = train.columns
#train = pd.DataFrame(train_imp, columns=train_columns)
#train = pd.concat([train,data_train_in.ix[:,'target']],axis=1)
#train.to_csv(train_filename,index=0)
#test_filename = (os.path.splitext(os.path.basename(sys.argv[2]))[0]+"_imm.csv")
#test_columns = test.columns
#test = pd.DataFrame(test_imp, columns=test_columns)
#if 'target' in data_test_in:
    #test = pd.concat([test,data_test_in.ix[:,'target']],axis=1)
#test.to_csv(test_filename,index=0)

#print train_filename, test_filename

log.close()


