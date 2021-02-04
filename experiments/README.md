# Experiments

- The experiments are structured in workflows
    - basic, ... (to be added)
- We implement three different approaches
    - baseline, advanced1, advanced2
- Our experiments can be run with five different models
    - resnet18, resnet50, resnet152, goolenet, mobilenet

## Run Experiments

### Requirements

- The different instances (server, node) are model as docker containers, thus docker needs to be installed to run them

### Preparation

- Navigate into the usecase's directory, for example: experiments/workflows/basic
- Create a copy of the file `setup/.env-template` and save it under `setup/.env`
- Set the fields in the newly created files (the paths you specify will be created)
    - SHARED_MOUNTED_DIR - The dir that will be mounted in all containers.
    - SERVER_MOUNTED_DIR - The dir that will be mounted in the server container.
    - NODE_MOUNTED_DIR - The dir that will be mounted in the node container. (If there are multiple nodes there might be
      multiple of these variables)
    - EVAL_MOUNTED_DIR - The dir that will be mounted in the evaluation container.

### Run

- Run the script `run_experiment.sh` with the following parameters
    - `-m` - to define the model to use
    - `-a` - to define the approach to use
    - `-l` - to define the location of the `.whl` file for the used `mmlib`(how to create the `.whl` file is described
      in the [mmlib repo](https://github.com/slin96/mmlib))

### What happens (depending on the chosen parameters)

- The specified in the `.env` directories are created and filled with content
- The docker containers are started (node(s), server, eval, mongoDB)
- The directories mounted, and a docker network will be established
- The experiment code is run in the node and the server containers
    - Logs from Python are written to `SHARED_MOUNTED_DIR/logs/<APPROACH>/python-<INSTANCE>.log`
    - Also files, such as stored models are stored under `SHARED_MOUNTED_DIR/logs/<APPROACH>/`
- The results and logs are evaluated by the eval container
    - the result of the evaluation is logged under `SHARED_MOUNTED_DIR/logs/<APPROACH>/python-eval.log`
