#!/bin/python

import pandas as pd
#import numpy as np

dv = pd.Series([ '4', '2', 'Z', 'Z', 'Z', 'Z'])

print dv

rdv = list(dv)
enum = list(set(dv))
print "rdv ", rdv, "    enum= ",enum
enum.sort() #set is unordered; could lead to problems of column mismatch later, so best to sort....
for x in range(0,len(dv)):
    for i in range(0,len(enum)):
        print "rdv[x] = ",rdv[x]," type: ", type(rdv[x]),"  enum[i]=",enum[i]," type: ", type(enum[i]), "  dv[x] = ",dv[x]," type: ", type(dv[x])
        if dv[x] == enum[i]:
            rdv[x] = i;
print rdv
