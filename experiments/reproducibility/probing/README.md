# Probing

- all experiments use the probing tool provided by the mmlib

## Probing Tool Figure
- the python script `probe_figure.py` was used to produce the output used to create the image of the probing tool 
  output in the thesis.
  
## Probing Models

- to probe all relevant models we use the script 
  `/Users/nils/Studium/master-thesis/repo/experiments/reproducibility/probing/probe_models.py`
- to show the reproducibility of the models, we execute the script on two nodes (we name them node1 and node2) with 
  different parameters

### Execution

- node1
    - on node1 we only save probing summaries
    - `probe_models.py --save_path <SAVE-PATH>`

- node2
    - on node2 we create summaries and compare them to the ones form node2
    - `probe_models.py --save_path <SAVE-PATH> --load_path <LOAD-PATH> --compare true`
    
#### Results
- **all models are reproducible across the machines we tested on**

- log of node1
    - `results/probe-models/node1-out.txt`
    
- log of node2
    - `results/probe-models/node1-out.txt`
    
- note: the probing tool color codes the output, to see display the output color coded open the files with the `less` 
  command
    - e.g. `less node1-out.txt`