# Probing Experiment

- all experiments use the probing tool provided by the mmlib

## TODO

- description of probing tool
- refer to mmlib docu

## Probing Models

- to probe all relevant models we use the script
  `experiments/reproducibility/probing/probe_models.py`
- we probe all our models in a CPU and a GPU environment

### CPU

- we create probing summaries on two different nodes (we refer to them as *node1* and *node2*)
- both nodes are part of the 16-node cluster
- because the created probing summaries can be very big, we only save the data for the first 10 and last 10 layers of a
  model

- having created the probe summaries we compare them to see of the results are reproducible

#### Execution

- **node1**
    - on node1 we only save probing summaries
    - `python probe_models.py --save_path <SAVE-PATH> > 16node1-out.txt`

- **node2**
    - on node2 we create summaries and compare them to the ones form node2
    - `python probe_models.py --save_path <SAVE-PATH> --load_path <LOAD-PATH> --compare true > 16node2-out.txt`

#### Results

- **all models are reproducible across the nodes we tested on**
- we can see this by taking a look in `16node2-out.txt`
    - the probing tool has a color coded output, to see it we can, for example, use the less
      command (`less 16node2-out.txt`)

### GPU

- we create probing summaries on the dgx-100 using two different (but same type) GPUs (we refer to them as *gpu1* and
  *gpu2*)
- because the created probing summaries can be very big, we only save the data for the first 10 and last 10 layers of a
  model

- having created the probe summaries we compare them to see of the results are reproducible

### Execution

- **gpu1**
    - we only save probing summaries
    - `CUDA_VISIBLE_DEVICES=0 python probe_models.py --save_path <SAVE-PATH> > gpu1-out.txt`

- **gpu2**
    - on node2 we create summaries and compare them to the ones form node2
    - `CUDA_VISIBLE_DEVICES=1 python probe_models.py --save_path <SAVE-PATH> --load_path <LOAD-PATH> --compare true >
      gpu2-out.txt`

#### Results

- **all models are reproducible across the gpus we tested on**
- we can see this by taking a look in `gpu2-out.txt`
    - the probing tool has a color coded output, to see it we can for example use the less command (`less gpu2-out.txt`)

## Probing Tool Figure

- the python script `probe_figure.py` was used to produce the output used to create the image of the probing tool output
  in the thesis.