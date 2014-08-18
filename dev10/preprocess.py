import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
import onehot

def preprocess(data):

    non_sparse_only = True
    use_all_category_only = False
    use_all_impute_mean_mode = False


    if non_sparse_only:
        nominal_samples = data.ix[:,['var4','dummy']] 
        onehot_samples = onehot.transform(nominal_samples,['var4','dummy'])
        onehot_samples = pd.DataFrame(onehot_samples.toarray())
        numbered_samples = data.ix[:,['var7','var8','var10','var11','var13','var15','var17']]
        numbered_samples[['var7','var8']] = numbered_samples[['var7','var8']].convert_objects(convert_numeric=True)
        #(var7 and 8 are ordinal, converting to floats which includes NaNs will allow mean imputing of missing values)
        other_samples = data.ix[:,'crimeVar1':'weatherVar236'] #all the continuous vars
        other_samples = other_samples.drop(['weatherVar115'], axis=1) #nothing in this feature
        samples = pd.concat([onehot_samples,numbered_samples,other_samples],axis=1) #combine w/ the cleaned up other vars
        imp_nan = Imputer(missing_values=np.nan, strategy='mean', axis=0)
        samples_imp = imp_nan.fit_transform(samples)
    
    if use_all_category_only:
        todo
    
    if use_all_impute_mean_mode:
        todo
    
    return samples_imp