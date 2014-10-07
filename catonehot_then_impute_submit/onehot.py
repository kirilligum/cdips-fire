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
        #print set(dv.replace(to_replace='Z',value=np.nan).dropna())
        #print [x for x in dv if x!='Z']
        #print set([x for x in dv if x!='Z'])
        if dv.isin(['Z']).any():
          enum = list(set([x for x in dv if x!='Z']))
        else:
          enum = list(set(dv))
        enum.sort() #set is unordered; could lead to problems of column mismatch later, so best to sort....
        #print"set_rdv", set(rdv),len(set(rdv)),"   enum=",enum
        #print "len(enum) = ", len(enum)
        #rdv = [enum.index(e) if e in enum else np.nan for e in dv]
        rdv.replace(to_replace=enum,value=range(len(enum)),inplace=True)
        #if rdv.str.contains('Z').any():
          #print set(rdv)
          #rdv[str(rdv)=='Z']=np.nan
        #print set(rdv)
        rdv.replace(to_replace='Z',value=np.nan,inplace=True)
        #print type(rdv)
        #print "set drv = ",set(list(set(rdv.tolist())))
        #print "set drv = ",set(rdv.tolist())
        #for x in range(0,len(rdv)):
            #for i in range(0,len(enum)):
                #if rdv.iloc[x] == enum[i]:
                    #rdv[x] = i
                #elif rdv.iloc[x] == 'Z':
                    #rdv[x] = np.nan
        #print"set_rdv after", set(rdv)
        return rdv
    def fit(self,data,cols):
      if 'var4' in cols and 'var4r' not in cols:
        data_var4r = data['var4'].str[0]
        data_var4r.name = 'var4r'
        data = pd.concat([data,data_var4r],axis=1)
        cols += ['var4r']
      self.cols=cols
      #print self.cols
      for i in cols:
        #print "set_data_i",set(data[i])
        d = self.col_to_list(data[i])
        #print "set_d",set(d)
        d.dropna(inplace=True)
        #print d.shape
        #print len(set(d))
        #print set([x for x in d.values.tolist()])
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
        for j,v in enumerate(d):
          if np.isnan(d[j]):
            col_nans += [1]
            d[j]=median
          else:
            col_nans += [0]
        if len(set(d))>1:
          tcols[i] = self.enc[i].transform([[x] for x in d.values.tolist()]).todense()
          for j,v in enumerate(tcols[i]):
            if col_nans[j]:
              tcols[i][j]=np.nan
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

