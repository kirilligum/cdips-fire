import pandas as pd

splits = []
#for i in range(5):
for i in range(50):
    if i<10:
        i= '0'+str(i)
    filename = 'b'+str(i)+'/test_oh_c96_imgmm_tr_xfr.csv'
    #print filename
    splits += [pd.read_csv(filename,index_col='id')]

#print [j.shape for j in splits]
combined = pd.concat(splits,1)
print combined.shape
ave = combined.mean(1)
ave.name='target'
ave.to_csv("ave_target.csv",header=1)


