# EMET [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**EMET** is a neural network used to classify fake social media posts based on information retrieved from traditional news source.

## More information

You can find more information in our paper [_EMET: Embeddings from Multilingual-Encoder Transformer for Fake News Detection_](https://ieeexplore.ieee.org/abstract/document/9054673).
The dataset used can be downloaded on [figshare](https://figshare.com/s/6aedbe9887ceeab64aea)

## Docker image

`docker build <emet/fakenewsDetection:latest> .`

	> sudo nvidia-docker run --tty --interactive --userns=host --volume /home/EMET:/home/usr/work --name EMET emet/fakenewsDetection:latest /bin/bash

## Run EMET

`python3 run_emet.py -t /dataset/paper_train.pkl -v /dataset/paper_test.pkl`

# Acknowledgment

Thank you [Jing Yang](https://github.com/Jinnab) for your collaboration in solve some mistakes on the network.