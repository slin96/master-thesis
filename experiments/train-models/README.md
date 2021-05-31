# Generate Models

## Setup

- in order to run the experiment *pytorch* and *torchvision* have to be installed
    - for example: `conda install pytorch=1.7.1 torchvision cudatoolkit=11.0 -c pytorch`
- also *mmlib* needs to be installed
    - for instructions see https://github.com/slin96/mmlib
- finally set the pythonpath
    - `export PYTHONPATH="<PATH-TO-REPO-ROOT>"`

## Training

- we trained the models using `train-model-imagenet.py`, with:
    - workers: 0
    - save-freq 2