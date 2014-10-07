 cat ../data/test.csv | cut -f 1,14 -d ',' > id_var13.csv
 tail -n +2 id_var13.csv | perl -F, -lane 'print $F[0],",",-1.0*$F[1]'>> byvar13.csv
