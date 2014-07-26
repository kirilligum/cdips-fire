import numpy as np
import pandas as pd
from __future__ import division

def random_bool(bs,n):
    #Randomly down-sample boolean series (bs) to n True entries
    if n > sum(bs):
        raise Exception("Cannot select more than total number of True entries")
    ind = np.nonzero(bs)[0] #get the indices
    ind_rand = random.sample(ind,n)
    new_bs = np.zeros(len(bs), dtype=bool) #initialize new boolean
    new_bs[ind_rand] = True                  #back to boolean
    new_bs = pd.Series(new_bs)          #necessary?
    return new_bs
            
def selector(data, n_train_fire, n_train_nofire, n_validate):
    """ Takes training DataFrame, # of fire and # of no-fire policies to train on
    and # of samples to validate on. As per Chris's suggestion, the validation set
    will have the same proportion of fires as the full set
    """
    had_fire = (data['target'] !=0)
    use_train = random_bool(had_fire,n_train_fire) | random_bool(~had_fire,n_train_nofire)
    n_validate_fire = int(n_validate*1188/452061)
    n_validate_nofire = n_validate - n_validate_fire
    if n_validate_fire + n_train_fire >1188:
        raise Exception("Too many fires! (validation set too large or number of training fires too large)")
    use_validate = random_bool((~use_train & had_fire), n_validate_fire) | \
                    random_bool((~use_train & ~had_fire), n_validate_nofire)
    return (use_train, use_validate)