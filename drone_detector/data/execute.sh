cd trimmed
python3 mkdir.py
cd test
python3 mkdir.py
cd ../../augmentation
python3 mkdir.py
cd ../csv
python3 mk_aug_csv.py
python3 mk_sec_csv.py
cd ../code
python3 extraction.py
python3 augmentation.py
cd ../augmentation
./copy.sh
cd ../code
python3 trimming.py
