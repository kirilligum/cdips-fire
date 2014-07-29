import pandas as pd
import numpy as np
import selector
import preprocess
reload(preprocess)
import gini
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import time
import pickle

#main2 is for testing RFClassifier (main is for RFRegressor)

fires_in_train = 924 #max is 1188-validation_size*(1188/452061)
nofires_in_train = 350000
validation_size = 100000
ntrees = 100
repeat = 10 #repeat model fit and validation this many times

tic = time.time()
data = pd.read_csv('C:\Users\Adam\Documents\FireLoss\cdips\\train.csv')
toc = time.time()
seconds_elapsed = toc - tic
print "Reading .csv took", seconds_elapsed, "seconds" 

tic = time.time()
samples_imp = preprocess.preprocess(data)
toc = time.time()
seconds_elapsed = toc - tic
print "preprocessing took", seconds_elapsed, "seconds" 

score=[]
RFscore=[]
for i in range(repeat):
    use_train, use_validate = selector.selector(data, fires_in_train, nofires_in_train, validation_size)
    
    samples = samples_imp[use_train,:]
    labels = data.ix[use_train,'target']
    
    labels = labels !=0

    weights = np.asarray(data.ix[use_train,'var11'])
    
    forest = RandomForestClassifier(n_estimators = ntrees,n_jobs=-1)
    
    tic = time.time()
    forest = forest.fit(samples,labels,weights)
    toc = time.time()
    seconds_elapsed = toc - tic
    print "RF fit took", seconds_elapsed, "seconds" 
    
    validate_samples = samples_imp[use_validate,:]
    predictions = forest.predict_proba(validate_samples)
    predictions = predictions[:,1]
    actual = data.ix[use_validate,'target']
    
    actual = actual !=0
    
    weights = data.ix[use_validate,'var11']

    predictions = predictions*1000 + weights/10000 #this is to order the zero values according to weight....
    
    score.append(gini.normalized_weighted_gini(actual,predictions,weights))
    print "fires in train: %d, records in RF fit: %d, of records in validation set: %d, ntrees = %d" % \
            (fires_in_train, fires_in_train+nofires_in_train, validation_size, ntrees)
    print "Normalized weighted Gini score = ", score[i]
            
    validate_weights=np.asarray(data.ix[use_validate,'var11'])
    RFscore.append(forest.score(validate_samples,actual,sample_weight=validate_weights))
    print "RF score = ", RFscore[i]
    
results = {'score': [score], 'RFscore': [RFscore],'ntrees':ntrees,'fires_in_train':fires_in_train, 
    'nofires_in_train':nofires_in_train,'validation_size':validation_size}
pickle.dump(results, open( "resultsClassif3.p", "wb" ) )
