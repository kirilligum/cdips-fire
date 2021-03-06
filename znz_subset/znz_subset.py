import pandas as pd
#import numpy as np
#import selector
#import preprocess
#import time
#import pickle
import random

#data = pd.read_csv('../data/train.csv',nrows=10000)
#data = pd.read_csv('../data/train.csv',nrows=100000)
data = pd.read_csv('../data/train.csv')
#data = pd.read_csv('train.csv')
print data.shape

zeros = []
nonzeros = []


target_data = data.ix[:,'target']

for index, row in target_data.iteritems():
    if row:
        nonzeros += [index]
    else:
        zeros += [index]

print len(nonzeros)
print len(zeros)
random.shuffle(nonzeros)
random.shuffle(zeros)

nonzeros = nonzeros[:100]
zeros = zeros[:900]

folds=10
z_incremental_subset_size = int(len(zeros)/folds);
print "z_incremental_subset_size = ",z_incremental_subset_size
nz_incremental_subset_size = int(len(nonzeros)/folds);
print "nz_incremental_subset_size = ",nz_incremental_subset_size

#cv_sets_nonzeros_indecies = []
#cv_sets_zeros_indecies = []

for i in range(folds):
  #cv_sets_nonzeros_indecies += nonzeros[nz_incremental_subset_size*i:nz_incremental_subset_size*(i+1)]
  #cv_sets_zeros_indecies += zeros[z_incremental_subset_size*i:z_incremental_subset_size*(i+1)]
  ifrom = nz_incremental_subset_size*i
  ito = nz_incremental_subset_size*(i+1)
  print "ifrom nz to ", ifrom, "  ", ito
  cv_sets_nonzeros_indecies_test = nonzeros[ifrom:ito]
  if i==0:
    cv_sets_nonzeros_indecies_train = nonzeros[ito:]
  elif i==folds-1:
    cv_sets_nonzeros_indecies_train = nonzeros[:ifrom]
  else:
    cv_sets_nonzeros_indecies_train = nonzeros[:ifrom]+nonzeros[ito:]
  print "len(nonzeros) ",len(nonzeros),"  ",len(cv_sets_nonzeros_indecies_test), "  ", len(cv_sets_nonzeros_indecies_train)

  ifrom = z_incremental_subset_size*i
  ito = z_incremental_subset_size*(i+1)
  print "ifrom z to ", ifrom, "  ", ito
  cv_sets_zeros_indecies_test = zeros[ifrom:ito]
  if i==0:
    cv_sets_zeros_indecies_train = zeros[ito:]
  elif i==folds-1:
    cv_sets_zeros_indecies_train = zeros[:ifrom]
  else:
    cv_sets_zeros_indecies_train = zeros[:ifrom]+zeros[ito:]
  print "len(zeros) ",len(zeros),"  ",len(cv_sets_zeros_indecies_test), "  ", len(cv_sets_zeros_indecies_train)

  cv_sets_nonzeros_indecies_train.sort()
  cv_sets_zeros_indecies_train.sort()
  cv_sets_nonzeros_indecies_test.sort()
  #print cv_sets_nonzeros_indecies_test
  cv_sets_zeros_indecies_test.sort()

  filename = ("znz_train_%d.csv" % (i))
  sub_data = data.iloc[cv_sets_nonzeros_indecies_train+cv_sets_zeros_indecies_train]
  sub_data.to_csv(filename,index=0)
  filename = ("znz_test_%d.csv" % (i))
  sub_data = data.iloc[cv_sets_nonzeros_indecies_test+cv_sets_zeros_indecies_test]
  #sub_data = data.iloc[cv_sets_nonzeros_indecies_test]
  sub_data.to_csv(filename,index=0)
