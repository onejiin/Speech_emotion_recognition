# Speech Emotion Recognition for Various model (CNN, RNN)

This code is based on "Hemanth Kumar Veeranki" speech-emotion-recognition (Clone to : https://github.com/hkveeranki/speech-emotion-recognition.git)

## Introduction

speech emotion recognition 4 models
1) CNN by Hemanth Kumar Veeranki
2) RNN by Hemanth Kumar Veeranki
3) CNN+RNN, Efficient Emotion Recognition from Speech Using Deep Learning on Spectrogram, Aharon Satt et al., Interspeech 2017
4) Bidirectional LSTM, On the use of Self-supervised Pre-trained Acoustic and Linguistic Features for Continuous Speech Emotion Recognition, Manon Macary et al., arxiv 2020

dataset : IEMOCAP data, 2280, 
Training : 1824(80%), Validation : 456(20%)

Validation performance
1) CNN - 51.3%
2) LSTM - 60.7%
3) CNN+LSTM - 64.3%
4) BiLSTM - 64.5%

## Dependencies

* Python 3.6.11
* tensorflow 2.2.0
* Keras 2.4.3
* h5py 2.10.0
* scipy 1.4.1
* sklearn 0.20.0
* speechpy 2.4

## Usage
### Download Dataset
#### Training and validataion database
Download and unzip [IEMOCAP](http://sail.usc.edu/iemocap/).  
I use IEMOCAP_full_release_withoutVideos.tar and unzip in IEMOCAP
#### Test EarningCall database
Download [EarningCall](https://github.com/GeminiLn/EarningsCall_Dataset).
(There is no emotion labels.)
 
### Train Model
```bash
$ python train.py
usage: train.py [--model_type choices=['cnn', 'rnn', 'inter2017', 'Manon2020']]
                [--save_model=Model Save Path]
                [--is_train]
                [--epoch=Epoch Num]
                [--select_test_data=['EMOCAP', 'EarningCall’]]
                [--EMOCAP_path=EMOCAP data location]
                [--EarningCall_path=EarningCall data location]
                [--load_model=In inferencing, model location]
```
For example,

1.) training
```bash
$ python train.py --model_type cnn --is_train --save_model ./models/cnn_train.h5 
                  --select_test_data EMOCAP --IEMOCAP_path ./IEMOCAP/IEMOCAP_full_release
                  --EarningCall_path ./EarningsCall_Dataset/Amazon_Inc_20170202/Wav
```
2.) inference
```bash
$ python train.py --model_type cnn  
                  --select_test_data EMOCAP --IEMOCAP_path ./IEMOCAP/IEMOCAP_full_release
                  --EarningCall_path ./EarningsCall_Dataset/Amazon_Inc_20170202/Wav
                  --load_model ./models/cnn_train.h5
```

## Contribution

I created a model architecture that accepts the same speech input format.
I add 2 models, CNN+RNN and Bidirectional LSTM. It improves performance comparing previous CNN and LSTM models