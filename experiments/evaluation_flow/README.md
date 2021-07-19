# Evaluating approaches on the Evaluation flow

- in the thesis we describe that we evaluate all our models on a *evaluation flow*
- in this directory you find all code and instructions to execute the experiments

- disclaimer
  - all the code to run the experiments is to certain extent tied to the setup we found at our uni machines
  - we do not guarantee that the code als works in other setup, still it can be seen as a good starting point for 
    reproducing pur results in another setup
    
## Preparation 
- we execute our experiments on three different but equivalent machines in the same cluster
  - one node is representing the *database*, on the *server*, and one the *node*
  
- for the *node* and the *server* we create corresponding directories using the scripts and copy them to the node and 
  the server respectively:
  - `create-node-dir.py`
  - `create-server-dir.py`

- to model the database we run a mongoDB in a container in the corresponding machine
- in our setup we use enroot
- before we can automatically run all experiments, we have to ensure that the command `enroot start mongo` will strat a 
  mongoDb container, to do si follow the following steps:
    - pull docker image
      - `enroot import docker://mongo`
    - create container sandbox
      - `enroot create --name mongo mongo.sqsh`
    - list the sandboxes 
      - `enroot list`
    - start container
      - `enroot start mongo`
  
- all our experiments make use of the approaches developed in the `mmlib` library
  - thus, it needs to be installed on the machine representing the `node` and the `server`
  - see instructions on how to install it in the corresponding [mmlib repo](https://github.com/slin96/mmlib)
    - *HINT*: you can also build the library locally and then just copy the wheel file to the corresponding machines
      and install it there using pip
  




# Evaluate the different mmlib approaches

- disclaimer
  - script `run-experiment.py` was used to run experiments on uni machines
  - it is not guaranteed that it works any other machine
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