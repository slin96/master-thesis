# dtoolAI: Reproducibility for Deep Learning

- suggest some specific guidelines for the generation and use of DL models in science
- refer to the FAIR priciples 
- develop a python package implementing these guidelines called: dtoolAI
- **mainly interesting for us**: 
    - Definitions on Reproducability, ...
    - might give some insights of how to track metadata in PyTorch: [GitHub Repo](https://github.com/JIC-CSB/dtoolai)


## Reproducibility: Concepts and Terminology
- corresponding to classification B1 in [Barba’s system](https://arxiv.org/pdf/1802.03311.pdf):
- **Reproducibility** is the ability to regenerate results using the original researchers’ data, software, and parameters.
- **Replicability** is the ability to arrive at the same result using new data.
- **Repeatability** is the ability to rerun a published analysis pipeline and arrive at the same results.

- their aim: want to repeat the generation of a DL model in order to reproduce its results
- due to random initializations they expect to only be able to guarantee reproducibility within a certain tolerance

## Provenance
- The provenance of a computational object is the history of the process used to produce it together with their input data
- this makes it key for reproducibility 

## ML and Reproducibility: Training
- model weights depend on training data and the parameters of the training process (hyperparameters)
- factors that we need to know in order to reproduce models:
    - loss function
    - type of optimizer (+ its hyperparameters)
    - learning rate 
    - how input is preprocessed (e.g. augmentation process)
    
## Guidelines for Reproducibility in DL 
- Annotate model training data with metadata
- give data persistent URIs 
- capture training parameters at model training time
- store training parameters and data inputs together with the model
- **comment Nils**: Besides the annotation of data with metadata and uri, this cold also be solved by having a well 
designed config in the experiment code, why do wh need all the boilerplate of dtoolAI?

    
    