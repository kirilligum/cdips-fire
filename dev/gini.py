#code and test case taken from:
#https://www.kaggle.com/c/liberty-mutual-fire-peril/forums/t/9685/will-a-python-script-for-the-evaluation-metric-be-made-available
from __future__ import division
import pandas as pd

def weighted_gini(act,pred,weight):
    df = pd.DataFrame({"act":act,"pred":pred,"weight":weight})    
    df.sort('pred',ascending=False,inplace=True)        
    df["random"] = (df.weight / df.weight.sum()).cumsum()
    total_pos = (df.act * df.weight).sum()
    df["cum_pos_found"] = (df.act * df.weight).cumsum()
    df["lorentz"] = df.cum_pos_found / total_pos
    df["gini"] = (df.lorentz - df.random) * df.weight  
    return df.gini.sum()

def normalized_weighted_gini(act,pred,weight):
    return weighted_gini(act,pred,weight) / weighted_gini(act,act,weight)

# should return -0.6813186813186815
#print normalized_weighted_gini([0,0,1,0,1],[.1,.4,.3,1.2,0],[1,2,5,4,3])