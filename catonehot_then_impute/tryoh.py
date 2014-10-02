#!/bin/python

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

l = [0,1,0,0]
print OneHotEncoder().fit_transform([l])
print "other"
print OneHotEncoder().fit_transform([[x] for x in l])
