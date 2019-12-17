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
pytorch
3. You will also need to download and install below 3 packages, 2 libraries after you install either python or anaconda from the steps above
   - pytorch 
   - librosa
   - numpy
   - scipy
   - tqdm
   
  - if you have chosen to install python 3.6 then run below commands in command prompt/terminal to install these packages
   ```
   pip3 install torch torchvision
   pip install librosa
   pip install numpy
   pip install scipy
   pip install tqdm
   ```
   - if you have chosen to install anaconda then run below commands in anaconda prompt to install these packages
   ```
   conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
   conda install -c conda-forge librosa
   conda install -c anaconda numpy
   conda install -c anaconda scipy
   conda install -c conda-forge tqdm
   ```   

## Usage

Explain how to run the automated tests for this system

### Preparing dataset

### Training a model
Explain what these tests test and why

```
Give an example
```

### Inferencing with a trained model

Explain what these tests test and why

```
Give an example
```


## Authors

* **Sungyoun Seo(Yun)** - *Audio data processing, Neural network models* - [pleiades-s](https://github.com/pleiades-s/)
* **Seunghyun Yeo** - *Audio data processing, Acoustic node*

## Acknowledgments

* [ksanjeevan](https://github.com/ksanjeevan/crnn-audio-classification) inspires us with its model architecture.
* [Junwon Hwang](https://github.com/nuxlear?tab=overview&org=keras-team&from=2018-12-01&to=2018-12-31) helps technical details for audio classification.
