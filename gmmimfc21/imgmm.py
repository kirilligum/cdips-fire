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
    self.mean_ranges = pd.DataFrame(columns=self.data.columns)
    self.g = mixture.GMM()
    return
  def fit(self,fit_data,min_covar=0.01, n_components=1):
    self.g = mixture.GMM(n_components=n_components,covariance_type='full', min_covar=min_covar)
    self.g.fit(fit_data)
  def transform(self,fit_data):
    decide_node_sample = random.random()
    iweight = 0
    running_sum = 0
    while decide_node_sample>running_sum:
      running_sum+=self.g.weights_[iweight]
      iweight+=1
    iweight-=1
    #self.mean_ranges.loc[len(self.mean_ranges)] = self.g.means_[iweight] #prints out mean ranges for each gmm run
    print >>log,fit_data.shape, self.g.means_[iweight].shape,self.mean_ranges.shape
    ### sample nans for each sample
    for isodx, impute_sample in fit_data.ix[:,:].iterrows():
      ### separate the vector into known and unknown.
      mis = [] # indecies of missing
      pis = [] # indecies of present (not missing)
      for idx,i in enumerate(impute_sample):
        if np.isnan(i):
          mis+=[idx]
        else:
          pis+=[idx]
      if mis:
        m_mean = self.g.means_[iweight][mis]
        p_mean = self.g.means_[iweight][pis]
        mm_cov = self.g.covars_[iweight][np.ix_(mis,mis)]
        mp_cov = self.g.covars_[iweight][np.ix_(mis,pis)]
        pm_cov = self.g.covars_[iweight][np.ix_(pis,mis)]
        pp_cov = self.g.covars_[iweight][np.ix_(pis,pis)]
        ### here we find conditional means and covariances
        mp_div_pp_cov = np.dot(mp_cov,np.linalg.inv(pp_cov))
        m_given_p_mean = m_mean + np.dot(mp_div_pp_cov,(impute_sample[pis]-p_mean))
        m_given_p_cov = mm_cov - np.dot(mp_div_pp_cov,pm_cov)
        impran= scipy.stats.multivariate_normal.rvs(m_given_p_mean,m_given_p_cov)
        #impran= scipy.stats.multivariate_normal.rvs(m_mean,mm_cov)
        #impran= scipy.stats.multivariate_normal.rvs(self.g.means_[iweight][mis],self.g.covars_[iweight][mis]*np.eye(len(mis)))
        for idx,i in enumerate(impute_sample[:]):
          if np.isnan(i):
            impute_sample[idx]=impran[0]
            impran=impran[1:]
        fit_data.iloc[isodx]=impute_sample
    return fit_data

log = open(os.path.splitext(sys.argv[0])[0]+'.log','wt')

data_train_in = pd.read_csv(sys.argv[1])
data_test_in = pd.read_csv(sys.argv[2])
train = data_train_in.drop('target',axis=1)
if 'target' in data_test_in:
  test = data_test_in.drop('target',axis=1)
else:
  test = data_test_in
data = pd.concat([train,test])
print >> log,  train.shape, "  ", test.shape ,"  " , data.shape

#data = pd.read_csv(sys.argv[1])
#miss = data.apply(lambda x : x.apply(lambda y: np.isnan(y))) # T/F data frame of missing values
imp_mean = Imputer(missing_values=np.nan, strategy='mean', axis=0)
fit_data = pd.DataFrame(imp_mean.fit_transform(data),columns=data.columns)
#np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
ig = impute_gmm(data)
for igmm_min_covar in [0.1]:
  ig.fit(fit_data,igmm_min_covar,2)
  train_imp = ig.transform(train)
  test_imp = ig.transform(test)

id_train = data_train_in.ix[:,'id']
id_test = data_test_in.ix[:,'id']
train_imp = pd.concat([id_train,train_imp],axis=1)
test_imp = pd.concat([id_test,test_imp],axis=1)

train_filename = (os.path.splitext(os.path.basename(sys.argv[1]))[0]+"_gmmim.csv")
train = pd.concat([train,data_train_in.ix[:,'target']],axis=1)
train.to_csv(train_filename,index=0)
test_filename = (os.path.splitext(os.path.basename(sys.argv[2]))[0]+"_gmmim.csv")
if 'target' in data_test_in:
    test = pd.concat([test,data_test_in.ix[:,'target']],axis=1)
test.to_csv(test_filename,index=0)

print train_filename, test_filename

log.close()


