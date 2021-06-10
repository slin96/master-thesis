# MMLIB Provenance Approach Experiment

- the [probing experiment](../probing) shows us that we can reproduce model training on CPUs and GPUs
- to show that the mmlib provenance approach is capable of saving and recovering models without loss of precision:
    - we train and save a *mobilenet* model in several steps and save it after every training as a new model on one node
    - we recover all models on another node and set `execute_checks=True` for the recover method to guarantee that the

## Setup & Execution

- in total, we use three nodes to run this experiment, all nodes have access to a shared file system
- all nodes are nodes of the 16-node cluster
- **DBnode**: runs an instance of mongoDB in a docker container, reachable for the two other nodes
- **node1**: runs the script `save_models.py` with the following parameters
    - `--training_data_path` specifying the path to the training data loaded by the dataloader
    - `--config` the path to the mmlib config (for more information on this see the mmlib repo)
    - `--tmp_dir` the directory to a directory that *node1* and *node2* can access via the shared file system
      (needs to be the same for *node1* and *node2*)
    - `--mongo_host` the ip or host name of the *DBnode*
- **node1**: runs the script `recover_models.py` with the following parameters
    - `--recover_ids` the ids to recover as a string formatted like this `"<ID1>,<ID2>,<ID2>"`
    - `--config` the path to the mmlib config (for more information on this see the mmlib repo)
    - `--tmp_dir` the directory to a directory that *node1* and *node2* can access via the shared file system
      (needs to be the same for *node1* and *node2*)
    - `--mongo_host` the ip or host name of the *DBnode*

## Results

- the logs of the executions can be found in the results directory
- the only important thing to note is that in `recover-log.txt` we can not find any assertion errors that would have
  been thrown if a saved and a recovered model differ 
  







