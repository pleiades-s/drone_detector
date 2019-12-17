## Description of folders

1. [*augmentation*](https://github.com/pleiades-s/drone_detector/tree/master/drone_detector/data/augmentation) folder for augmented data

2. [*code*](https://github.com/pleiades-s/drone_detector/tree/master/drone_detector/data/code) folder has extraction and trimming code for learning set test set. Also it has augmentation code for leraning set.

2. [*csv*](https://github.com/pleiades-s/drone_detector/tree/master/drone_detector/data/csv) folder has mk_aug_csv.py an  mk_sec_csv.py files which are for the csv files that describing relation between trimmed data and label.
 
3. [*rawdata*](https://github.com/pleiades-s/drone_detector/tree/master/drone_detector/data/rawdata) folder has original data. So you have to download data to here from this ([Link](https://drive.google.com/open?id=1Ywlhga3Ak7Ep54mcfuoQ35kijVbK5aWU)) 

4. [*trimmed*](https://github.com/pleiades-s/drone_detector/tree/master/drone_detector/data/trimmed) folder has test folder and each has 10 folders from 0.1 to 1.0


**You can simply prepare dataset with cmd bleow**
  
```
./execute.sh
```
