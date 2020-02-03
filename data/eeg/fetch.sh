# Get training data.
wget https://archive.ics.uci.edu/ml/machine-learning-databases/eeg-mld/SMNI_CMI_TRAIN.tar.gz
tar xzf SMNI_CMI_TRAIN.tar.gz
mv SMNI_CMI_TRAIN train
find train | grep gz$ | xargs gunzip

# Get test data.
wget https://archive.ics.uci.edu/ml/machine-learning-databases/eeg-mld/SMNI_CMI_TEST.tar.gz
tar xzf SMNI_CMI_TEST.tar.gz
mv SMNI_CMI_TEST test
find test | grep gz$ | xargs gunzip
