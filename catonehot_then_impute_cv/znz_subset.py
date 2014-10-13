import pandas as pd
#import numpy as np
#import selector
#import preprocess
#import time
#import pickle
import random
import sys
import os


log = open(os.path.splitext(sys.argv[0])[0]+'.log','wt')

if not sys.stdin.isatty():
    inputfile = sys.stdin
    outputfile = sys.argv[0]
    if outputfile[-3:]=='.py':
        outputfile = outputfile[:-3]

elif len(sys.argv) > 1:
    inputfile = sys.argv[1]
    outputfile = sys.argv[0]
    if outputfile[-3:]=='.py':
        outputfile = outputfile[:-3]
    outputfile = outputfile + "_" + os.path.splitext(inputfile)[0]
    print >> log,  "input: ", inputfile
    print >> log,  "output", outputfile
    if len(sys.argv)>2:
        outputfile = sys.argv[2]
        print >> log,  "output", outputfile
else :
    print >> log,  "no inputs"
    inputfile = 'train.csv'
    outputfile = sys.argv[0]
#inputfile = open(sys.stdin).read()
data = pd.read_csv( inputfile)
print >> log,  data.shape

zeros = []
nonzeros = []


target_data = data.ix[:,'target']

for index, row in target_data.iteritems():
    if row:
        nonzeros += [index]
    else:
        zeros += [index]

print >> log,  len(nonzeros)
print >> log,  len(zeros)
random.shuffle(nonzeros)
random.shuffle(zeros)

#nonzeros = nonzeros[:100]
#zeros = zeros[:900]
#zeros = zeros[:len(nonzeros)*3]

folds=10
z_incremental_subset_size = int(len(zeros)/folds);
print >> log,  "z_incremental_subset_size = ",z_incremental_subset_size
nz_incremental_subset_size = int(len(nonzeros)/folds);
print >> log,  "nz_incremental_subset_size = ",nz_incremental_subset_size

#cv_sets_nonzeros_indecies = []
#cv_sets_zeros_indecies = []

test_files = []
train_files = []

for i in range(folds):
  #cv_sets_nonzeros_indecies += nonzeros[nz_incremental_subset_size*i:nz_incremental_subset_size*(i+1)]
  #cv_sets_zeros_indecies += zeros[z_incremental_subset_size*i:z_incremental_subset_size*(i+1)]
  ifrom = nz_incremental_subset_size*i
  ito = nz_incremental_subset_size*(i+1)
  print >> log,  "ifrom nz to ", ifrom, "  ", ito
  cv_sets_nonzeros_indecies_test = nonzeros[ifrom:ito]
  if i==0:
    cv_sets_nonzeros_indecies_train = nonzeros[ito:]
  elif i==folds-1:
    cv_sets_nonzeros_indecies_train = nonzeros[:ifrom]
  else:
    cv_sets_nonzeros_indecies_train = nonzeros[:ifrom]+nonzeros[ito:]
  print >> log,  "len(nonzeros) ",len(nonzeros),"  ",len(cv_sets_nonzeros_indecies_test), "  ", len(cv_sets_nonzeros_indecies_train)

  ifrom = z_incremental_subset_size*i
  ito = z_incremental_subset_size*(i+1)
  print >> log,  "ifrom z to ", ifrom, "  ", ito
  cv_sets_zeros_indecies_test = zeros[ifrom:ito]
  if i==0:
    cv_sets_zeros_indecies_train = zeros[ito:]
  elif i==folds-1:
    cv_sets_zeros_indecies_train = zeros[:ifrom]
  else:
    cv_sets_zeros_indecies_train = zeros[:ifrom]+zeros[ito:]
  print >> log,  "len(zeros) ",len(zeros),"  ",len(cv_sets_zeros_indecies_test), "  ", len(cv_sets_zeros_indecies_train)

  cv_sets_nonzeros_indecies_train.sort()
  cv_sets_zeros_indecies_train.sort()
  cv_sets_nonzeros_indecies_test.sort()
  #print >> log,  cv_sets_nonzeros_indecies_test
  cv_sets_zeros_indecies_test.sort()

  filename = (outputfile+"_train_%d.csv" % (i))
  sub_data = data.iloc[cv_sets_nonzeros_indecies_train+cv_sets_zeros_indecies_train]
  sub_data.to_csv(filename,index=0)
  train_files += [filename]
  filename = (outputfile+"_test_%d.csv" % (i))
  sub_data = data.iloc[cv_sets_nonzeros_indecies_test+cv_sets_zeros_indecies_test]
  #sub_data = data.iloc[cv_sets_nonzeros_indecies_test]
  sub_data.to_csv(filename,index=0)
  test_files += [filename]

log.close()

outputfile_name = open(os.path.splitext(sys.argv[0])[0]+'_prefix.txt','wt')

output_info = outputfile + "_f" + str(folds)

for train, test in zip(train_files, test_files):
    print train,test
    print >> outputfile_name, train,test

#print >> outputfile_name, output_info
#print output_info
