#!/bin/python

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class onehot:
    def __init__(self):
        self.values = {}
        self.columns = {}
        self.enc = {}
        self.parents = []
    def col_to_list(self,dv):
        rdv = dv
        #rdv = list(dv)
        if 'Z' in dv:
          enum = list(set(dv.remove('Z')))
        else:
          enum = list(set(dv))
        enum.sort() #set is unordered; could lead to problems of column mismatch later, so best to sort....
        for x in range(0,len(rdv)):
            for i in range(0,len(enum)):
                if rdv.iloc[x] == enum[i]:
                    rdv[x] = i
                elif rdv.iloc[x] == 'Z':
                    rdv[x] = np.nan
        return rdv
    def var4root(self,col):
      cdv = []
      if not self.parents:
          self.parents = [x[0] for x in list(set(col)) if len(x)>1 and int(x[1])==2] # truncating to the first letter and remove letters with only one number child
          self.parents.sort()
      cdv = [x[0] if (self.parents.count(x[0])) else 0 for x in col] # truncating to the first letter and remove letters with only one number child
      return self.col_to_list(cdv)
    def fit(self,data,cols):
      if 'var4' in cols and 'var4r' not in cols:
        data_var4r = data['var4'].str[0]
        data_var4r.name = 'var4r'
        data = pd.concat([data,data_var4r],axis=1)
        cols += ['var4r']
      self.cols=cols
      #print self.cols
      for i in cols:
        d = self.col_to_list(data[i])
        d.dropna(inplace=True)
        #print d.shape
        #print len(set(d))
        if len(set(d))>1:
          enc =  OneHotEncoder().fit([[x] for x in d.values.tolist()])
          self.enc[i]= enc
          #print self.enc
          self.values[i] = [enc.n_values_]
      #print self.values
    def transform(self,data):
      if 'var4' in data.columns and 'var4r' not in data.columns:
        data_var4r = data['var4'].str[0]
        data_var4r.name = 'var4r'
        data = pd.concat([data,data_var4r],axis=1)
      data = data[self.cols]
      #print self.cols, data.columns
      tcols = {}
      nans ={}
      combine = []
      for i in self.cols:
        d = self.col_to_list(data[i])
        col_nans = []
        col_nans_value = []
        nonan = d.dropna()
        median = nonan.median()
        #print "median", median
        for j,v in enumerate(d):
          #print j
          if d[j] is np.nan:
            col_nans += [1]
            d[j]=median
          else:
            col_nans += [0]
        #print i, col_nans
        #print self.enc
        if len(set(d))>1:
          #print self.enc[i]
          #print self.enc[i].transform([[x] for x in d.values.tolist()])
          tcols[i] = self.enc[i].transform([[x] for x in d.values.tolist()]).todense()
          for j,v in enumerate(tcols[i]):
            if col_nans[j]:
              tcols[i][j]=np.nan
          #print i,tcols[i].shape
          #print i, [i+"_"+str(x) for x in range(tcols[i].shape[1])]
          ddd = pd.DataFrame(tcols[i],columns = [i+"_"+str(x) for x in range(tcols[i].shape[1])])
          combine += [ddd]
          #combine += [pd.DataFrame(tcols[i].todense(), columns = [i+x for x in tcols[i].)]
        else:
          for j,v in enumerate(d):
            if col_nans[j]:
              d[j]=np.nan
          combine += [d]
          #print i,d.shape
      #print type(pd.DataFrame(tcols.values()[0].todense()))
      #print pd.concat([pd.DataFrame(x.todense()) for x in tcols.values()],axis=1)
      #print "combine"
      #print combine
      #print pd.concat(combine,axis=1)
      return pd.concat(combine,axis=1)

