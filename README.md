# Single Node Detection on Direction of Approach(DOA)

We have worked on detecting the UAV’s DOA with a single node, which consists of two acoustic sensors and a Raspberry Pi, to overcome an acoustic sensor limitation through combining machine learning mechanisms (CNN, CRNN, ResNet50).

## Table of Contents

- [Paper and presentation material](#paper-and-presentation-material)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
  * [Preparing dataset](#preparing-dataset)
  * [Training a model](#training-a-model)
  * [Inferencing with a trained model](#inferencing-with-a-trained-model)
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)

## Paper and presentation material
All details of this study are explained on materials listed below.
* [Paper](https://drive.google.com/open?id=1hsRaTrgqBGXRUGHPR9OH9D8tZbOcwYst "paper link")

* [Presentation material](https://drive.google.com/open?id=1gxR1evrUhUNqPMxn_eckKDMqFEDJ8CZC "presentation link")

## Prerequisites

What things you need to install the software and how to install them:

1. Python 3.6 
   - This setup requires that your machine has python 3.6 installed on it. you can refer to this url https://www.python.org/downloads/ to download python. Once you have python downloaded and installed, you will need to setup PATH variables (if you want to run python program directly, detail instructions are below in *how to run software section*). To do that check this: https://www.pythoncentral.io/add-python-to-path-python-is-not-recognized-as-an-internal-or-external-command/.  
   - Setting up PATH variable is optional as you can also run program without it and more instructon are given below on this topic. 
   
2. Second and easier option is to download anaconda and use its anaconda prompt to run the commands. To install anaconda check this url https://www.anaconda.com/download/

3. You will also need to download and install below 3 packages, 2 libraries after you install either python or anaconda from the steps above
   - pytorch 
   - librosa
   - numpy
   - scipy
   - tqdm
   
  - If you have chosen to install python 3.6 then run below commands in command prompt/terminal to install these packages
   ```
   pip3 install torch torchvision
   pip install librosa
   pip install numpy
   pip install scipy
   pip install tqdm
   ```
   - If you have chosen to install anaconda then run below commands in anaconda prompt to install these packages, libraries
   ```
   conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
   conda install -c conda-forge librosa
   conda install -c anaconda numpy
   conda install -c anaconda scipy
   conda install -c conda-forge tqdm
   ```   

## Usage

### Preparing dataset
You can use dataset collected in person. There are 3 stages for preparing dataset. 
  1. Extract UAV's audio from entire recorded audio file (wav format). 
  2. Augment the extracted audio file. 
  3. Split the augmented audio file into various length.
  
Those steps will be automatically executed by using the command below so you can use our dataset for train and test.
```
./execute.sh
```
You can download UAV's raw audio data into [rawdata](https://github.com/pleiades-s/drone_detector/tree/master/drone_detector/data/rawdata) folder under [data](https://github.com/pleiades-s/drone_detector/tree/master/drone_detector/data) folder via this link ([Download](https://drive.google.com/open?id=1Ywlhga3Ak7Ep54mcfuoQ35kijVbK5aWU))


### Training a model

We have worked on three neural network models.

- CNN1D: [cnn1d.py](https://github.com/pleiades-s/drone_detector/blob/master/drone_detector/code/cnn1d.py)
- CRNN: [cnn-lstm.py](https://github.com/pleiades-s/drone_detector/blob/master/drone_detector/code/cnn-lstm.py)
- ResNet50: [transfer_learning.py](https://github.com/pleiades-s/drone_detector/blob/master/drone_detector/code/transfer_learning.py)

There are three different data length(0.1, 0.5, 1.0 second) as a input size so length arguments are required. 
(*0.1, 0.5 and 1.0 are ONLY available*)
If the arguments are more than one, then the models will be triained in a row.

You can train a model with a command below
```
python3 [model] [input length]
```

* Example
```
python3 cnn1d.py 0.1 0.5 1.0

python3 cnn-lstm.py 0.1

python3 transfer_learning.py 1.0 0.5 0.1
```

### Inferencing with a trained model

You can inference trained model with [*inference.py*](https://github.com/pleiades-s/drone_detector/blob/master/drone_detector/code/inference.py) on test set.

You can train a model with a command below
```
python3 inference.py [trained model(.pth format)]
```

* Example
```
python3 inference.py ../model/Conv_Lstm_mfcc_1_2019-11-19_19:50:20.pth
```

## Authors

* **Sungyoun Seo([pleiades-s](https://github.com/pleiades-s/))** - *Audio data processing, Neural network models*
* **Seunghyun Yeo([yeoseunghyun](https://github.com/yeoseunghyun))** - *Audio data processing, Acoustic node*

## Acknowledgments

* [ksanjeevan](https://github.com/ksanjeevan/crnn-audio-classification) inspires us with its model architecture.
* [Junwon Hwang](https://github.com/nuxlear?tab=overview&org=keras-team&from=2018-12-01&to=2018-12-31) helps technical details for audio classification.

**If you have any question, please email me or write an issue. Any questions are welcome :)**
