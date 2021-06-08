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
    - on node 1 we only save probing summaries
    - `probe_models.py --save_path <SAVE-PATH>`

- node2
    - on node2 we create summaries and compare them to the ones form node2
    - `probe_models.py --save_path <SAVE-PATH> --load_path <LOAD-PATH> --compare true`
    

#### Results
- LINK TO OUTPUT FORM 16-NODE cluster