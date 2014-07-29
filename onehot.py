#!/bin/python

from sklearn.preprocessing import OneHotEncoder

def col_to_list(dv):
    enum = list(set(dv))
    enum.sort() #set is unordered; could lead to problems of column mismatch later, so best to sort....
#     if enum.count('Z'):
#         enum.insert(len(enum),enum.pop(enum.index('Z'))) #move Z (missing value) to be the last element
    for x in range(0,len(dv)):
        for i in range(0,len(enum)):
            if dv[x] == enum[i]:
                dv[x] = i;
    return dv


def transform(data,cols): # takes pandas.read_csv and list of strings of variables, and returns onehot matrix
    d =[]
    for i in cols:
        dv= data.ix[:,i]
        if i=='var4': # taking care of categories by making another label of just the letter (label[0])
            cdv = []
            cdv = [x[0] if (len(x)>1 and int(x[1])>1) else 'X' for x in dv] # truncating to the first letter and remove letters with only one number child
            #this adds in dummy X's, but I believe they are necessary as place-holders
            #(previous version of onehot.transform did not seem to work for var4)
            #can possibly remove 'X' column later?
            d.append(col_to_list(cdv))
        d.append(col_to_list(dv))
    enc = OneHotEncoder()
    enc.fit(zip(*d))
    td = enc.transform(zip(*d))
    return td

