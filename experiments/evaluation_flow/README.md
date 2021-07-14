# Evaluate the different mmlib approaches

- disclaimer
  - script ____ was used to run experiments on uni machines
  - not guaranteed on any other machine
  - but script can be good starting point to adjust for new environment

- need to have ssh access to the machines 
  - with a ssh key identification

- install enroot
  - mongoDB

- install conda
  - install mmlib in named env
  
- models need to be trained and ready
  - we produced them like THIS
  - can be downloaded here

- data needs to be trained and ready
  - we used this ____ data
  - can be downloaded here REF to DATA dir in repo

## Models for experiment

## relation
```
U1 ---> U3_1_1 ---> U3_1_2 ---> ...
|
|-> U2 ---> U3_2_1 ---> U3_2_2 ---> ...
```

- U1 model
    - models use case 1
    - is the pretrained model given by PyTorch
- U2 model 
    - models use case 2
    - take U1 as starting point
    - train for 10 epochs on the imagenet validation dataset
- U3_1 models
    - models use case 3
    - first takes model U1 as starting point
    - following take their previous U3_1 model as starting point
    - trains on 512 images of ONE category of the Custom Coco dataset
    - are trained for 5 epochs in comparison to the previous model
- U3_2 models
    - same as U3_1 models but initially take U2 as starting point