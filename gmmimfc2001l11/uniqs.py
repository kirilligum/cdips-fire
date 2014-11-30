import pandas as pd
import sys
import os
import re

df = pd.read_csv(sys.argv[1])
print df.shape

for i in df.columns:
  print i, len(df[i].unique())
