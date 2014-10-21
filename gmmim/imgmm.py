import pandas as pd
import numpy as np
import scipy
from sklearn.preprocessing import Imputer
from sklearn import mixture
import time
import sys
import os
import random
import itertools

log = open(os.path.splitext(sys.argv[0])[0]+'.log','wt')

data = pd.read_csv(sys.argv[1])
#data = pd.read_csv(sys.argv[1]).iloc[:10,[1]]
print >> log, "data.shape = " , data.shape

miss = data.apply(lambda x : x.apply(lambda y: np.isnan(y)))

imp_mean = Imputer(missing_values=np.nan, strategy='mean', axis=0)
data_imp_mean = pd.DataFrame(imp_mean.fit_transform(data),columns=data.columns)
print data_imp_mean.shape, data.shape
#print >> log, data_imp_mean, data

g = mixture.GMM(n_components=1, min_covar=0.1,thresh=0.001)
#g = mixture.GMM(n_components=1, covariance_type='full',min_covar=0.5)
g.fit(data_imp_mean)
print 'weights_ = ',g.weights_.shape
print 'means_ = ',g.means_.shape,type(g.means_)
print 'covars_ = ',g.covars_.shape,type(g.covars_)
print >> log, 'weights_ = ',g.weights_.shape
print >> log, 'weights_ = ',g.weights_
#print 'means_ = ',g.means_.shape
print >> log, 'means_ = ',g.means_.shape
print >> log, 'means_ = ',g.means_
#print 'covars_ = ',g.covars_.shape
print >> log, 'covars_ = ',g.covars_.shape
print >> log, 'covars_ = ',g.covars_
#print 'converged_ = ',g.converged_
print >> log, 'converged_ = ',g.converged_

### Decide from which gaussian to sample by sampling from weights
decide_node_sample = random.random()
print decide_node_sample , g.weights_
iweight = 0
running_sum = 0
while decide_node_sample>running_sum:
  running_sum+=g.weights_[iweight]
  iweight+=1
  print "running_sum = ", running_sum
iweight-=1
print iweight
print 'means_ = ',g.means_[iweight][:50]

### separate the vector into known and unknown.

imps=[]
prefs =[]
for isodx, impute_sample in data.ix[:,:].iterrows():
  nan_means = []
  nan_covars = []
  val_means = []
  val_covars = []
  impute_sample_nan_idxs = []
  impute_sample_num_idxs = []
  for idx,i in enumerate(impute_sample):
    if np.isnan(i):
      impute_sample_nan_idxs+=[idx]
    else:
      impute_sample_num_idxs+=[idx]
  print >>log,impute_sample_nan_idxs
  #print impute_sample_nan_idxs ,impute_sample_num_idxs,data.columns[impute_sample_nan_idxs]
  #print impute_sample_num_idxs, impute_sample[impute_sample_num_idxs]
  print >>log,impute_sample_num_idxs
  #impran= np.random.multivariate_normal(g.means_[iweight][impute_sample_nan_idxs],g.covars_[iweight][np.ix_(impute_sample_nan_idxs,impute_sample_nan_idxs)])
  if impute_sample_nan_idxs:
    impran= scipy.stats.multivariate_normal.rvs(g.means_[iweight][impute_sample_nan_idxs],g.covars_[iweight][impute_sample_nan_idxs]*np.eye(len(impute_sample_nan_idxs)))
    #impran= scipy.stats.multivariate_normal.rvs(g.means_[iweight][impute_sample_nan_idxs],g.covars_[iweight][np.ix_(impute_sample_nan_idxs,impute_sample_nan_idxs)])
  else:
    impran=1
  #print impran
  #print >> log, "pref norm =", g.means_[iweight][impute_sample_num_idxs]
  #print >> log, "pref cov =",g.covars_[iweight][impute_sample_num_idxs]
  #print >> log, "pref cov =",g.covars_[iweight][np.ix_(impute_sample_num_idxs,impute_sample_num_idxs)]
  #print "num =",impute_sample[impute_sample_num_idxs]
  #print "means = ",g.means_[iweight][impute_sample_num_idxs]
  #print "covs = ",g.covars_[iweight][np.ix_(impute_sample_num_idxs,impute_sample_num_idxs)]
  #pref = scipy.stats.multivariate_normal.pdf(impute_sample[impute_sample_num_idxs],g.means_[iweight][impute_sample_num_idxs],g.covars_[iweight][impute_sample_num_idxs]*np.eye(len(impute_sample_num_idxs)))
  #prefs+=[pref]
  #print pref
  #impran*= pref
  imps+=[impran]
  print >>log,isodx," imputed = ",impran
#print >>log,"sorted(pref): ",np.sort(prefs)
#print "sort(pref): ",np.sort(prefs)[-10:]

#print imps



### calculate the coefficient from the known

## sample multivariate normal with uknown.


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


