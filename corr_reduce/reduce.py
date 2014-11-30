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


log = open(os.path.splitext(sys.argv[0])[0]+'.log','w')

corr = pd.read_csv(sys.argv[1])
print >> log,  corr.shape

print corr.shape


delcols = []

#corr = corr.ix[89:98,89:98]

current_cols = corr.columns
#print current_cols

corr_thresh = 0.99

while len(current_cols) > 0:
  icol = current_cols[0]
  correlated = corr[corr[icol]>corr_thresh].index.tolist()
  #print corr[icol]
  #print 'corr is one',corr[corr[icol]==1]
  if len(correlated)>1:
    #print 'correlated',correlated, corr.columns[correlated]
    delcols.extend(correlated[1:])
    #print delcols
  current_cols = current_cols.drop(current_cols[0])
  #print current_cols
print >>log,len(sorted(set(delcols)))
#print corr.columns[sorted(set(delcols))].tolist()

  #for idx,row in corr.iterrows():
    #print type(row) , row.shape, corr[idx].name

open("reduce_"+str(int(100*corr_thresh))+"_"+os.path.splitext(os.path.basename(sys.argv[1]))[0]+".txt",'w').write("\n".join( corr.columns[sorted(set(delcols))].tolist()))
open("reduce_num_"+str(int(100*corr_thresh))+"_"+os.path.splitext(os.path.basename(sys.argv[1]))[0]+".txt",'w').write("\n".join( sorted(set(str(delcols)))))
#corr.to_csv(corr_filename,index=0)
#print corr_filename


log.close()


