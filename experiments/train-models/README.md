# Generate Models

## Setup

- in order to run the experiment *pytorch* and *torchvision* have to be installed
    - for example: `conda install pytorch=1.7.1 torchvision cudatoolkit=11.0 -c pytorch`
- also *mmlib* needs to be installed
    - for instructions see https://github.com/slin96/mmlib
- finally set the pythonpath
    - `export PYTHONPATH="<PATH-TO-REPO-ROOT>"`

## Training on Imagenet Validation data


- **mobilenet**
    - ```
      python train-model-imagenet.py
        --num-epochs 51
        --save-root <PATH> 
        --workers 0 
        --save-freq 2
        --imagenet-root <IMAGENET-PATH>
        --model mobilenet 
        > <PATH>/out.txt
      ```
      
- **resnet18**
    - ```
      python train-model-imagenet.py
        --num-epochs 51
        --save-root <PATH> 
        --workers 0 
        --save-freq 2
        --imagenet-root <IMAGENET-PATH>
        --model resnet18 
        > <PATH>/out.txt
      ```
      
- **resnet50**
    - ```
      python train-model-imagenet.py
        --num-epochs 51
        --save-root <PATH> 
        --workers 0 
        --save-freq 2
        --imagenet-root <IMAGENET-PATH>
        --model resnet50 
        > <PATH>/out.txt
      ```
      
- **resnet152**
    - ```
      python train-model-imagenet.py
        --num-epochs 51
        --save-root <PATH> 
        --workers 0 
        --save-freq 2
        --imagenet-root <IMAGENET-PATH>
        --model resnet152 
        > <PATH>/out.txt
      ```
      
- **googlenet**
    - ```
      python train-model-imagenet.py
        --num-epochs 51
        --save-root <PATH> 
        --workers 0 
        --save-freq 2
        --imagenet-root <IMAGENET-PATH>
        --model googlenet 
        > <PATH>/out.txt
      ```