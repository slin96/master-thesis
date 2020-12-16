# Repeatability Experiments

## Manual Setup

- create a dir to execute the experiments in, e.g. `tmp`
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
  `master-thesis/experiments/repeatability/scripts /setup.sh`)
  
## Experiment: Repeatable Inference

### 1. Create Output on the first Node

- double check that the venv under `master-thesis/experiments/venv` is activated
    - `source master-thesis/experiments/venv/bin/activate`

- set python path
    - `export PYTHONPATH="<PATH>/master-thesis/"`

- cd into `master-thesis/experiments/repeatability/inference`
- execute `python3 output.py --imagenet-root <IMAGENET-ROOT> --tmp-output-root <OUTPUT-ROOT> --number-batches <BATCHES>`

### 2. Create Output on the second Node

- repeat the steps form 1. (maybe also the setup), and make sure
    - you are on a different node
    - set `<OUTPUT-ROOT>` to another dir to not overwrite the output

### 3. Compare Outputs

- if necessary:
    - copy the results written to `<OUTPUT-ROOT>`to ONE node
    - activate the `venv` and set `PYTHONPATH` (see above)
- execute `python3 compare.py --input-root <INPUT-ROOT> --compare-to-root <COMPATE-TO-ROOT>`
    - `<INPUT-ROOT>` and `<COMPATE-TO-ROOT>` are the `<OUTPUT-ROOT>`s from steps 1. and 2.

#### Results

If the results of step 1. and 2. are the same, step 3. should output: `ALL OUTPUTS ARE THE SAME`