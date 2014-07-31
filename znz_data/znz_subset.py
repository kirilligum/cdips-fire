import pandas as pd
#import numpy as np
#import selector
#import preprocess
#import time
#import pickle
import random

#data = pd.read_csv('../data/train.csv',nrows=10000)
#data = pd.read_csv('../data/train.csv',nrows=100000)
data = pd.read_csv('../data/train.csv')
#data = pd.read_csv('train.csv')
print data.shape

zeros = []
nonzeros = []


target_data = data.ix[:,'target']

for index, row in target_data.iteritems():
    if row:
        nonzeros += [index]
    else:
        zeros += [index]

random.shuffle(nonzeros)
nonzeros =  nonzeros[:1100]
nonzeros.sort()
random.shuffle(zeros)
zeros =  zeros[:8900]
zeros.sort()

sub_data = data.iloc[zeros+nonzeros]

sub_data.to_csv('znz_subset.csv')
