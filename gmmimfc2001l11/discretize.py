import pandas as pd
import os
import sys

data = pd.read_csv(sys.argv[1])
continuous = data.loc[:,'var10':'weatherVar236'].columns
discrete_nominal_orig = [ x+"_" for x in ['var2','var4','var5','var6','var9','dummy']]
discrete_nominal = [x for x in data.columns if "_" in x]
discrete_ordinal = ['var1','var3','var7','var8']
discrete = discrete_nominal+discrete_ordinal
#print continuous
data.drop(continuous,axis=1,inplace=1)
data.drop(discrete,axis=1,inplace=1)
#data = data.drop['var10:weatherVar236']
print data.columns
# get possible value

# for  nominal (>0.5)?1:0

# for ordinal  pick closest to an integer
