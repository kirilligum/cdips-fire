#module load anaconda
#cat znz_subset_prefix.txt | while read -r line; do python prep.py $line; done > znz10fold_prep_prefix.log
head -n1000 raw_train.csv > small_train.csv
head -n1000 raw_test.csv > small_test.csv
python onehot_transform.py small_train.csv small_test.csv 
python impute_mean_mean.py small_train_ohtrans.csv small_test_ohtrans.csv 
python rfr.py small_train_ohtrans_imm.csv small_test_ohtrans_imm.csv 
