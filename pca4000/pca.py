import pandas as pd
from sklearn.decomposition import PCA
import time
import sys
import os


log = open(os.path.splitext(sys.argv[0])[0]+'.log','wt')

data_train_in = pd.read_csv(sys.argv[1])
data_test_in = pd.read_csv(sys.argv[2])
train = data_train_in.drop(['target','id'],axis=1)
if 'target' in data_test_in:
  test = data_test_in.drop(['target','id'],axis=1)
else:
  test = data_test_in.drop('id',axis=1)
data = pd.concat([train,test])
print >> log,  train.shape, "  ", test.shape ,"  " , data.shape

print >> log, " starting pca"
start = time.clock()
pca = PCA()
pca.fit(data)
pca_train = pca.transform(train)
pca_test = pca.transform(test)
with open('components_.txt','wt') as f:
  print >> f, pca.components_
with open('explained_variance_ratio_.txt','wt') as f:
  print >> f, pca.explained_variance_ratio_
with open('mean_.txt','wt') as f:
  print >> f, pca.mean_
with open('n_components_.txt','wt') as f:
  print >> f, pca.n_components_
with open('noise_variance_.txt','wt') as f:
  print >> f, pca.noise_variance_
end = time.clock()
print >> log, "time = ", end-start


suffix = '_pca.csv'
train_filename = (os.path.splitext(os.path.basename(sys.argv[1]))[0]+suffix)
train = pd.DataFrame(pca_train)
train = pd.concat([data_train_in.ix[:,'target'],train],axis=1)
train = pd.concat([data_train_in.ix[:,'id'],train],axis=1)
train.to_csv(train_filename,index=0)
test_filename = (os.path.splitext(os.path.basename(sys.argv[2]))[0]+suffix)
test = pd.DataFrame(pca_test)
if 'target' in data_test_in:
  test = pd.concat([data_test_in.ix[:,'target'],test],axis=1)
test = pd.concat([data_test_in.ix[:,'id'],test],axis=1)
test.to_csv(test_filename,index=0)

print train_filename, test_filename

log.close()
