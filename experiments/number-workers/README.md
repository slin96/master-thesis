# Experiment: Number of Workers

## Reproducibility

- Does the number of workers for the dataloader affect the reproducibility? -- **YES**
- The number of workers has to be the same, to guarantee a reproducible execution

### Experiment

- to show the statement above we execute four training runs in different configurations
- two with 0 workers -- meaning the data will be loaded in the main process
- two with 5 workers -- meaning the data will be loaded using 5 subprocesses

#### Setup

- in order to run the experiment *pytorch* and *torchvision* have to be installed
    - for example: `conda install pytorch=1.7.1 torchvision cudatoolkit=11.0 -c pytorch`
- also *mmlib* needs to be installed
    - for instructions see https://github.com/slin96/mmlib
- finally set the pythonpath
    - `export PYTHONPATH="<PATH-TO-REPO-ROOT>"`
- to create the directories to save the models to run
    - `./create-directories.sh`

#### Dataset

- TODO

#### Run Experiment

- to perform the dummy training we use the script: `experiments/trained-models/train-model-imagenet.py`
- for our experiment we use the four different paramatrizations
    - **0 workers, run 1**
        - ```
          python train-model-imagenet.py
            --num-epochs 3 
            --save-root <LOCAL-ABS-PATH>/workers-0-run-1 
            --workers 0 
            --save-freq 1
            --imagenet-root /Users/nils/Studium/master-thesis/repo/tmp/imgnet 
            --model mobilenet 
            --num-batches 10 
            > <LOCAL-ABS-PATH>/workers-0-run-1/out.txt
          ```

    - **0 workers, run 2**
        - ```
            python train-model-imagenet.py
            --num-epochs 3
            --save-root <LOCAL-ABS-PATH>/workers-0-run-2
            --workers 0
            --save-freq 1
            --imagenet-root <IMAGENET-PATH>
            --model mobilenet
            --num-batches 10 
            > <LOCAL-ABS-PATH>/workers-0-run-2/out.txt 
          ```

    - **5 workers, run 1**
        - ```
          python train-model-imagenet.py
            --num-epochs 3
            --save-root <LOCAL-ABS-PATH>/workers-5-run-1
            --workers 5
            --save-freq 1
            --imagenet-root <IMAGENET-PATH>
            --model mobilenet
            --num-batches 10 
            > <LOCAL-ABS-PATH>/workers-5-run-1/out.txt
          ```

    - **5 workers, run 2**
        - ```
          python train-model-imagenet.py
            --num-epochs 3
            --save-root <LOCAL-ABS-PATH>/workers-5-run-2
            --workers 5
            --save-freq 1
            --imagenet-root <IMAGENET-PATH>
            --model mobilenet
            --num-batches 10 
            > <LOCAL-ABS-PATH>/workers-5-run-2/out.txt 
          ```

#### Results

- for the results see the jupyter notebook [here](./reproducible.ipynb)

#### Environment

- torch_version='1.7.1'
- macOS 10.15.7 (x86_64)
- 2,8 GHz Quad-Core Intel Core i7
- 16 GB RAM
  




