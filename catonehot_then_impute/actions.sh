cat znz_subset_prefix.txt | while read -r line; do python prep.py $line; done > znz10fold_prep_prefix.log
