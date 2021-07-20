# Deterministic Execution Experiment

This experiment investigates what effect an enforced deterministic execution of a training routine has in terms of 
runtime.

## Setup
- in order to run the experiment *pytorch* and *torchvision* have to be installed
  - for example: `conda install pytorch=1.7.1 torchvision cudatoolkit=11.0 -c pytorch`
- also *mmlib* needs to be installed 
  - for instructions see https://github.com/slin96/mmlib
- finally set the pythonpath
  - `export PYTHONPATH="<PATH-TO-REPO-ROOT>"`

## Run Experiment
- the experiment code is defined in `time_training.py`
- to specify the GPUs used use something like
    - `CUDA_VISIBLE_DEVICES=1,2 python myscript.py`
- for our experiment we use two different parametrizations
  - **non-deterministic:** `python time_training.py --num-epochs <EPOCHS> --coco-root <COCO-ROOT> --coco-annotations <COCO-META-JSON> --model <MODEL-ID> > <LOG-FILE>`
  - **deterministic:** `python time_training.py --num-epochs <EPOCHS> --coco-root <COCO-ROOT> --coco-annotations <COCO-META-JSON> --model <MODEL-ID> --deterministic t > <LOG-FILE>`
  
## Environment
- we executed all experiments on the dgx-100 (for more info see main experiment README.md) using 1 GPU
- all our experiments that lead to the results in `./results` were executed in the following environment:
  - torch_version='1.7.1'
  - torchvision='0.8.2'
  - cuda_compiled_version='11.0'
  - os='Ubuntu 20.04.1 LTS (x86_64)'
  - nvidia_gpu=GPU 0: A100-SXM4-40GB
- we trained for 50 epochs
- we trained on our custom coco dataset, available here: https://owncloud.hpi.de/s/TRCzfvxwyHCRIQr
  
## Results
- the results are plotted in the jupyter notebook `plots.ipynb`
- the saved plots can be found under `./plots`