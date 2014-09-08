#!/bin/python
import pandas as pd
#import numpy as np
#import time


filename = ("0/predicted.csv")
#tmp = pd.read_csv(filename)
#datas = pd.DataFrame(tmp['target'],index=tmp.id)
datas = pd.read_csv(filename)

for i in range(1,100):
    filename = ("%d/predicted.csv" % (i))
    data_tmp = pd.read_csv(filename)
    #tmp = pd.read_csv(filename)
    #data_tmp = pd.DataFrame(tmp['target'],index=tmp.id)
    print data_tmp.shape
    datas = pd.merge(datas,data_tmp,on='id')
    #datas = datas.append(tmp)
#print len(datas), datas[0].shape
#dfdatas = pd.concat([datas],axis=1)

print datas.shape
#print datas

avedatas = datas.drop('id',axis=1).mean(axis=1)
ad = pd.concat({'id':datas.id,'target':avedatas},axis=1)
print ad.shape

filename = ("all.csv")
datas.to_csv(filename,index=0)
filename = ("ave.csv")
ad.to_csv(filename,index=0)
