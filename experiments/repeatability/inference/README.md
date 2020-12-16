# Experiments: Repeatable Inference

## Manual execution
### 1. Setup
- create a dir to execute the xperiments in, e.g. `tmp`
- clone the master thesis repo 
    - `git clone https://github.com/slin96/master-thesis.git`
- clone the mmlib repo 
    - `git clone https://github.com/slin96/mmlib.git`
- you should end up with something like: 
```
├── tmp/
│   ├── master-thesis/
│   ├── mmlib/
```
- now run the script: `sh setup.sh` (can be found under:
  `master-thesis/experiments/repeatability/inference/scripts /setup.sh`)

### 2. Create output on first Node
- double check that the venv under `master-thesis/experiments/venv` is activated
  - `source master-thesis/experiments/venv/bin/activate`
  
- set python path
  - `export PYTHONPATH="<PATH>/master-thesis/"`
  
- cd into `master-thesis/experiments/repeatability/inference`
- execute `python3 output.py --imagenet-root <IMAGENET-ROOT> --tmp-output-root <OUTPUT-ROOT> --number-batches <BATCHES>`

### 3. Create output on second Node
- repeat the steps form 2. (maybe also step 1.), and make sure
  - you are on a different node
  - set `<OUTPUT-ROOT>` to another dir to not overwrite the output 
  
### 4. Compare outputs
- if necessary:
  - copy the results written to `<OUTPUT-ROOT>`to ONE node
  - activate the `venv` and set `PYTHONPATH` (see above)
- execute `python3 compare.py --input-root <INPUT-ROOT> --compare-to-root <COMPATE-TO-ROOT>`
  - `<INPUT-ROOT>` and `<COMPATE-TO-ROOT>` are the `<OUTPUT-ROOT>`s from steps 2 and 3
  
### Results 
If the results of step 2. and 3. are the same, step 4. should output: `ALL OUTPUTS ARE THE SAME`