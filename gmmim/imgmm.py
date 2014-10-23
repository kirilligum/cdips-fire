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

class impute_gmm:
  def __init__(self,data):
    self.data = data
    self.nan_not_counted = True
    self.nans_and_nums=[]
    self.mean_ranges = pd.DataFrame(columns=self.data.columns)
    return
  def sample(self,fit_data,min_covar):
    g = mixture.GMM(n_components=8, min_covar=igmm)
    g.fit(fit_data)
    ### Decide from which gaussian to sample by sampling from weights
    decide_node_sample = random.random()
    iweight = 0
    running_sum = 0
    while decide_node_sample>running_sum:
      running_sum+=g.weights_[iweight]
      iweight+=1
    iweight-=1
    self.mean_ranges.loc[len(self.mean_ranges)] = g.means_[iweight]
    print fit_data.shape, g.means_[iweight].shape,self.mean_ranges.shape
    ### sample nans for each sample
    for isodx, impute_sample in self.data.ix[:,:].iterrows():
      nan_means = []
      nan_covars = []
      val_means = []
      val_covars = []
      ### separate the vector into known and unknown.
      if self.nan_not_counted:
        impute_sample_nan_idxs = []
        impute_sample_num_idxs = []
        for idx,i in enumerate(impute_sample):
          if np.isnan(i):
            impute_sample_nan_idxs+=[idx]
          else:
            impute_sample_num_idxs+=[idx]
        self.nans_and_nums += [impute_sample_nan_idxs,impute_sample_num_idxs]
        if isodx is self.data.ndim-1:
          self.nan_not_counted=False
      if impute_sample_nan_idxs:
        impran= scipy.stats.multivariate_normal.rvs(g.means_[iweight][impute_sample_nan_idxs],g.covars_[iweight][impute_sample_nan_idxs]*np.eye(len(impute_sample_nan_idxs)))
        for idx,i in enumerate(impute_sample[:]):
          if np.isnan(i):
            impute_sample[idx]=impran[0]
            impran=impran[1:]
        fit_data.iloc[isodx]=impute_sample
    return fit_data


log = open(os.path.splitext(sys.argv[0])[0]+'.log','wt')
data = pd.read_csv(sys.argv[1])
miss = data.apply(lambda x : x.apply(lambda y: np.isnan(y))) # T/F data frame of missing values
imp_mean = Imputer(missing_values=np.nan, strategy='mean', axis=0)
fit_data = pd.DataFrame(imp_mean.fit_transform(data),columns=data.columns)
#np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
ig = impute_gmm(data)
for igmm in [0.01]:
  fit_data = ig.sample(fit_data,igmm)
#print ig.mean_ranges.ix[:,:4]

data_filename = (os.path.splitext(os.path.basename(sys.argv[1]))[0]+"_imgmm.csv")
fit_data.to_csv(data_filename,index=0)
#fit_data.to_csv("gmm_impued.csv",index=0)

#print train_filename, test_filename

log.close()


