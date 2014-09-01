#!/bin/python
import pandas as pd
#import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
#from sklearn.metrics import accuracy_score
from bootstrap_znz import bootstrap_znz
from sklearn import svm
from sklearn.cross_validation import KFold
#from gini import normalized_weighted_gini

def get_nz(target_train,data_train):
    zeros = []
    nonzeros = []
    for index, row in target_train.iteritems():
        if row:
            nonzeros += [index]
        else:
            zeros += [index]
    nz_target_train = target_train.drop(zeros)
    nz_data_train = data_train.drop(zeros)
    return nz_target_train,nz_data_train

filename = ("train.csv")
#filename = ("../data/oh_imp_train.csv")
data_train = pd.read_csv(filename)
target_train = data_train['target']
data_train.drop('target',axis=1,inplace=True)
filename = ("test.csv")
#filename = ("../data/oh_imp_test.csv")
data_test = pd.read_csv(filename)

nz_target_train,nz_data_train = get_nz(target_train,data_train)

target_train[target_train!=0]=1
data_train.drop('target',axis=1,inplace=True)
#target_test[target_test!=0]=1
#data_test.drop('target',axis=1,inplace=True)

target_train,data_train = bootstrap_znz(target_train,data_train)

start = time.clock()

###############################################################
###### begin model

###### do classification
rfc= RandomForestClassifier(n_jobs=-1)
#rfc= RandomForestClassifier(n_estimators = 1000,n_jobs=-1)
rfc = rfc.fit(data_train,target_train)
predict_loc_class = pd.DataFrame({"target":rfc.predict(data_test), "id": data_test['id']})
nz_target_test,nz_data_test = get_nz(predict_loc_class['target'],data_test)
###### do multimodel regression
## stacking regressor
strfs = []
## make a list of models
rf = RandomForestRegressor()
#rf = RandomForestRegressor(n_jobs=-1)
svm = svm.SVR()
models = [rf,svm]

#### cv train of stackers
## create folds
kf = KFold(len(nz_data_train), n_folds=10,shuffle=True)
for train, test in kf:
## fit models
    for i in models:
        i.fit(nz_data_train.iloc[train],nz_target_train.iloc[train])
## predict
    stpredict = []
    for i in models:
        stpredict += [ pd.Series(i.predict(nz_data_train.iloc[test]))]
    dfstpredict = pd.concat(stpredict,axis=1)
## fit stackers
    strf = RandomForestRegressor()
    strf.fit(dfstpredict,nz_target_train.iloc[test])
    strfs += [strf]

#### stack
## fit all models
for i in models:
    i.fit(nz_data_train,nz_target_train)
#print models
## predict from all models
stpredict = []
for i in models:
    #tmp = pd.DataFrame({'target':i.predict(nz_data_test),'id':nz_data_test['id']})
    tmp = pd.DataFrame({models.index(i):i.predict(nz_data_test)},index=nz_data_test.index)
    #print nz_data_test
    #print tmp
    stpredict += [ tmp]
    #stpredict += [ pd.Series(i.predict(nz_data_test))]
dfstpredict = pd.concat(stpredict,axis=1)
dfstpredict = pd.concat([dfstpredict,nz_data_test['id']],axis=1)
## predict and average stackers
print dfstpredict
#print dfstpredict.mean(axis=1)
stack_predicted_list = []
for i in strfs:
    tmp = pd.DataFrame({strfs.index(i):i.predict(dfstpredict.drop('id',axis=1))},index=nz_data_test.index)
    #tmp = pd.Series(i.predict(dfstpredict.drop('id',axis=1)))
    stack_predicted_list += [tmp]
#print stack_predicted_list
stack_predicted = pd.concat(stack_predicted_list,axis=1).mean(axis=1)
predict_loc_regres = pd.DataFrame({"target":stack_predicted, "id": nz_data_test['id']})
print predict_loc_regres

#rfr = RandomForestRegressor(n_jobs=-1)
#rfr= RandomForestRegressor(n_estimators = 1000,n_jobs=-1)
#rfr = rfr.fit(nz_data_train,nz_target_train)
#predict_loc_regres = pd.DataFrame({"target":rfr.predict(nz_data_test), "id": nz_data_test['id']})

###### end model
###############################################################

end = time.clock()

#### merge regressed and 0s while preserving the order
predicted = pd.DataFrame(columns=predict_loc_class.columns)
for i in predict_loc_class['id']:
    if predict_loc_class[predict_loc_class.id==i].target.any()!=0:
        tt  = predict_loc_regres[predict_loc_regres.id==i].target
        predicted = predicted.append([pd.DataFrame({'id':int(i),'target':tt})])
    else:
        tt  = predict_loc_class[predict_loc_class.id==i].target
        predicted = predicted.append([pd.DataFrame({'id':int(i),'target':tt})])
predicted[['id']] = predicted[['id']].astype(int)

filename = ("predicted.csv")
predicted.to_csv(filename,index=0)




#predict = [a*b for a,b in zip(predict_loc_class,predict_loc_regres)]
#correct_targets += [float(len(target_test[target_test==1])-len(predict_loc_class[predict_loc_class==1]))/len(predict_loc_class)]
#training_score = rfc.score(data_train,target_train)
#score = rfr.score(nz_data_test,nz_target_test)

#rf_times += [end-start]
#predicts_class += [predict_loc_class]
#predicts += [predict]
#scores +=[score]
##training_scores +=[training_score]
#gn = normalized_weighted_gini(target_test,predict_loc_class,data_test.var11)
#gn = normalized_weighted_gini(target_test,predict,data_test.var11)
#ginis +=[gn]

#print "rf_times",np.array(rf_times).mean(),"x",len(rf_times)
##print "training_scores: ",training_scores, pd.DataFrame(training_scores).mean()
#print "correct_targets = " , np.array(correct_targets).mean(), np.array(correct_targets).std(), np.array(correct_targets).ptp()
#print "score: ", np.array(scores).mean(), np.array(scores).std(), np.array(scores).ptp()
#print "gini: ", np.array(ginis).mean(), np.array(ginis).std(), np.array(ginis).ptp()

