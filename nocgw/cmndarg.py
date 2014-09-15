#!/usr/bin/python

import sys
import pandas as pd

if len(sys.argv) > 1:
    print "input: ",sys.argv[1]
    if len(sys.argv)>2:
        print "output", sys.argv[2]
elif not sys.stdin.isatty():
    data = pd.read_csv(sys.stdin)
    print data.shape
else :
    print "no inputs"
