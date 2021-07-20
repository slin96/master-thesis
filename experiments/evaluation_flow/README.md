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
  
### Copy code
- for the *node* and the *server* we create corresponding directories using the scripts and copy them to the node and 
  the server respectively:
  - `create-node-dir.py`
  - `create-server-dir.py`

### Database
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
  
### MMlib
- all our experiments make use of the approaches developed in the `mmlib` library
  - thus, it needs to be installed on the machine representing the `node` and the `server`
  - see instructions on how to install it in the corresponding [mmlib repo](https://github.com/slin96/mmlib)
    - *HINT*: you can also build the library locally and then just copy the wheel file to the corresponding machines
      and install it there using pip
  
### Pretrained model
- to enable an extensive evaluation we do only perform dummy training (U2) or do not perform training at all (U3)
- instead, of training we load snapshots of models that we have trained before using the code in
  - [train-models-experiment](../train-models)
  - depending on our experimental setup we can vary the save frequency
- having created the model snapshots we have to bring them in the following directory structure so that we can access 
  them when executing our experiments
  - `<MODELNAME>-version-<U3DATASET>`
      - `use-case-1.pt` 
      - `use-case-2.pt`
      - `use-case-3-1-1.pt`
      - `use-case-3-1-2.pt`
      - `use-case-3-1-3.pt`
      - ...
      - `use-case-3-2-1.pt`
      - `use-case-3-2-2.pt`
      - `use-case-3-2-3.pt`
      - ...
  - `<MODELRELATION> - [mobilenet, googlenet, resnet18, resnet50, resnet152]`
  - `<U3DATASET> - [food, outdoor]`
  
- *COMMENT*: we plan to upload the exact models we used, but so far we do not have enough cloud storage available
  
### Data
- for use case 1 we use the pretrained modes provided by PyTorch
- for use case 2 we use the imagenet validation dataset
- for use cases 3 we use our custom created COCO-food or COCO-outdoor dataset
- for detailed descriptions on the datasets see the [data](../../data) part of this repo
  


## Run evaluation
- after preparing all the models, datasets, and the code we can run our experiments using the script
  - `run-experiment.py`
  
## Environments

### 16-Node cluster
- each machine
  - Fujitsu RX2530 M5
  - 96 GB RAM
  - 2 * Intel Xeon Gold 5220S, 18 Cores

### dgx-100
- NVIDIA DGX A100
- 1 TB RAM
- 8 * NVIDIA A100-SXM4-40GB