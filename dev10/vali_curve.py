import pandas as pd
import numpy as np
#from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import time
#import matplotlib.pyplot as plt
from sklearn.learning_curve import validation_curve

print "loading data"
start = time.clock()
data = pd.read_csv('../prep/prep.csv')
#print list(data.columns.values)
#print data.shape
#x, y  = data.ix[10:20,'1':'9'], data.ix[20:30,'target']
x, y  = data.ix[:,'0':'49'].as_matrix(), data.ix[:,'target'].as_matrix()
y = np.rint(y)
print type(x)
print x.shape
print type(y)
print y.shape[0]
print y
#print x
#print y
end = time.clock()
print "time = ", end-start


indices = np.arange(y.shape[0])
np.random.shuffle(indices)
x,y = x[indices],y[indices]

print" starting valid"
start = time.clock()
forest = RandomForestClassifier(n_jobs=-1)
#forest = RandomForestClassifier(n_estimators = 100,n_jobs=-1)
train_score, validation_score = validation_curve(forest,x,y,'n_estimators',np.arange(2,2+100*10,100),cv=5)
end = time.clock()
print "time to train = ", end-start

print train_score
np.savetxt("train_score.csv",train_score,delimiter=",")
print validation_score
np.savetxt("validation_score.csv",validation_score,delimiter=",")

train_score_mean = np.mean(train_score,axis=1)
validation_score_mean = np.mean(validation_score,axis=1)

print train_score_mean
np.savetxt("train_score_mean.csv",train_score_mean,delimiter=",")
print validation_score_mean
np.savetxt("validation_score_mean.csv",validation_score_mean,delimiter=",")
