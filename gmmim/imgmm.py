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
fit_data = pd.DataFrame(imp_mean.fit_transform(data),columns=data.columns)
print fit_data.shape, data.shape
#print >> log, fit_data, data
#print fit_data.ix[:0,:5]

nan_not_counted = True
nans_and_nums=[]
g = mixture.GMM(n_components=10, min_covar=0.001)

for igmm in range (3):
  g.fit(fit_data)
  ### Decide from which gaussian to sample by sampling from weights
  decide_node_sample = random.random()
  iweight = 0
  running_sum = 0
  while decide_node_sample>running_sum:
    running_sum+=g.weights_[iweight]
    iweight+=1
  iweight-=1
  ### sample nans for each sample
  for isodx, impute_sample in data.ix[:,:].iterrows():
    nan_means = []
    nan_covars = []
    val_means = []
    val_covars = []
    ### separate the vector into known and unknown.
    if nan_not_counted:
      impute_sample_nan_idxs = []
      impute_sample_num_idxs = []
      for idx,i in enumerate(impute_sample):
        if np.isnan(i):
          impute_sample_nan_idxs+=[idx]
        else:
          impute_sample_num_idxs+=[idx]
      print >>log,impute_sample_nan_idxs ,impute_sample_num_idxs
      nans_and_nums += [impute_sample_nan_idxs,impute_sample_num_idxs]
      nan_not_counted=False
    if impute_sample_nan_idxs:
      impran= scipy.stats.multivariate_normal.rvs(g.means_[iweight][impute_sample_nan_idxs],g.covars_[iweight][impute_sample_nan_idxs]*np.eye(len(impute_sample_nan_idxs)))
      print "total nans = ",len([j for j in impute_sample if np.isnan(j)])
      for idx,i in enumerate(impute_sample[:]):
        if np.isnan(i):
          print len(impran)
          impute_sample[idx]=impran[0]
          impran=impran[1:]
      fit_data.iloc[isodx]=impute_sample
  #print igmm,fit_data.ix[:0,:5].values


data_filename = (os.path.splitext(os.path.basename(sys.argv[2]))[0]+"_imgmm.csv")
fit_data.to_csv(data_filename,index=0)

#print train_filename, test_filename

log.close()


