#!/bin/python
import pandas as pd
#import numpy as np
import time
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.metrics import accuracy_score
#from sklearn.metrics import r2_score
#from gini import normalized_weighted_gini
#import random
import sys
import os


log = open(os.path.splitext(sys.argv[0])[0]+'.log','wt')

data_train_in = pd.read_csv(sys.argv[1])
data_test_in = pd.read_csv(sys.argv[2])
print >> log,  data_train_in.shape, "  ", data_test_in.shape

start = time.clock()
target_train = data_train_in['target']
data_train = data_train_in.drop(['id','target'],axis=1)
data_test = data_test_in.drop('id',axis=1)
if 'target' in data_test:
  target_test = data_test_in['target']
  data_test.drop('target',axis=1,inplace=True)

catcol = data_train_in.filter(like='var').columns.tolist()
mix_test_list = []
mix_train_list = []

start_gbr_tr = time.clock()
gbr = GradientBoostingRegressor(n_estimators =10,max_depth=7)
gbr_tr = gbr.fit(data_train,target_train)
transformed_train_gbr = gbr_tr.transform(data_train,threshold="0.35*mean")
print >> log, 'transformed_train_gbr',transformed_train_gbr.shape
transformed_test_gbr = gbr_tr.transform(data_test,threshold="0.35*mean")
gbr_tr_fit = GradientBoostingRegressor(n_estimators =10,max_depth=7)
gbr_tr_fit = gbr_tr_fit.fit(transformed_train_gbr,target_train)
mix_test_list += [pd.Series(gbr_tr_fit.predict(transformed_test_gbr),index=data_test_in.id.astype(int),name='gbr_tr')]
mix_train_list += [pd.Series(gbr_tr_fit.predict(transformed_train_gbr),index=data_train_in.id.astype(int),name='gbr_tr')]
end_gbr_tr = time.clock()
print >> log, "time_gbr_tr = ", end_gbr_tr-start_gbr_tr

start_xfr_tr = time.clock()
xfr= ExtraTreesRegressor(n_estimators =10,max_depth=7)
xfr_tr = xfr.fit(data_train,target_train)
transformed_train_xfr = xfr_tr.transform(data_train,threshold="0.35*mean")
print >> log, 'transformed_train_xfr',transformed_train_xfr.shape
transformed_test_xfr = xfr_tr.transform(data_test,threshold="0.35*mean")
xfr_tr_fit = ExtraTreesRegressor(n_estimators =10,max_depth=7)
xfr_tr_fit = xfr_tr_fit.fit(transformed_train_xfr,target_train)
mix_test_list += [pd.Series(xfr_tr_fit.predict(transformed_test_xfr),index=data_test_in.id.astype(int),name='xfr_tr')]
mix_train_list += [pd.Series(xfr_tr_fit.predict(transformed_train_xfr),index=data_train_in.id.astype(int),name='xfr_tr')]
end_xfr_tr = time.clock()
print >> log, "time_xfr_tr = ", end_xfr_tr-start_xfr_tr

start_gbr_cat = time.clock()
gbr_cat_fit = GradientBoostingRegressor(n_estimators =10,max_depth=7)
gbr_cat_fit = gbr_cat_fit.fit(data_train[catcol],target_train)
mix_test_list += [pd.Series(gbr_cat_fit.predict(data_test[catcol]),index=data_test_in.id.astype(int),name='gbr_cat')]
mix_train_list += [pd.Series(gbr_cat_fit.predict(data_train[catcol]),index=data_train_in.id.astype(int),name='gbr_cat')]
end_gbr_cat = time.clock()
print >> log, "time_gbr_cat = ", end_gbr_cat-start_gbr_cat

start_xfr_cat = time.clock()
xfr_cat_fit = ExtraTreesRegressor(n_estimators =10,max_depth=7)
xfr_cat_fit = xfr_cat_fit.fit(data_train[catcol],target_train)
mix_test_list += [pd.Series(xfr_cat_fit.predict(data_test[catcol]),index=data_test_in.id.astype(int),name='xfr_cat')]
mix_train_list += [pd.Series(xfr_cat_fit.predict(data_train[catcol]),index=data_train_in.id.astype(int),name='xfr_cat')]
end_xfr_cat = time.clock()
print >> log, "time_xfr_cat = ", end_xfr_cat-start_xfr_cat

start_xfr_mix = time.clock()
mix_test = pd.concat(mix_test_list,1)
mix_train = pd.concat(mix_train_list,1)
xfr_mix = ExtraTreesRegressor(n_estimators =10,max_depth=5)
xfr_mix.fit(mix_train,target_train)
with open('feature_importances_.txt','wt') as f:
  print >> f, xfr_mix.feature_importances_
xfr_mix_predict = pd.Series(xfr_mix.predict(mix_test),index=data_test_in.id.astype(int),name='target')
end_xfr_mix = time.clock()
print >> log, "time_xfr_mix = ", end_xfr_mix-start_xfr_mix

start_mix_ave = time.clock()
mix_ave = mix_test.mean(1)
mix_ave.name='target'
end_mix_ave = time.clock()

end = time.clock()

#if 'target_test' in locals():
  #score = xfr.score(data_test,target_test)
  #gn = normalized_weighted_gini(target_test,predict_loc_regres,data_test.var11)
#end = time.clock()

out_filename = (os.path.splitext(os.path.basename(sys.argv[2]))[0])
mix_test.to_csv(out_filename+"_mixmeth.csv")
mix_ave.to_csv(out_filename+"_mixave.csv",header=1)
xfr_mix_predict.to_csv(out_filename+'_mixxfr.csv')
print out_filename

log.close()
