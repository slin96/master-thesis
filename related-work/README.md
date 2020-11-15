# Related Work

## Categories 
- Here all the related work that is potential relevant is listed and categorized.
- A Paper of reference can occur in more than one category
- If the reference is no link but plane text they haven't been reviewed and can be seen as a TODO

### Terms and Definitions 
#### Machine/Deep Learning
- [Google ML Glossary](https://developers.google.com/machine-learning/glossary#validation)
- [Deep Learning Book](https://www.deeplearningbook.org)

#### Recoverability/Reproducibility
- [dtoolAI: Reproducibility for Deep Learning](#dtoolAI-Reproducibility-for-Deep-Learning)
- [Terminologies for reproducible research](#reproduce-vs-replicate-discussion)
- [ACM: Artifact Review and Badging](#reproduce-vs-replicate-discussion)
- Book: Reproducibility and replicability in science

### Schema definition
- [Automatically Tracking Metadata and Provenance of Machine Learning Experiments](#Automatically-Tracking-Metadata-and-Provenance-of-Machine-Learning-Experiments)
- [ML-Schema](#ML-Schema)
    
### Distributed Training/Infrastructure
- [Scalable Deep Learning on Distributed Infrastructures](#Scalable-Deep-Learning-on-Distributed-Infrastructures)
- [The Missing Piece in Complex Analytics: Low Latency, Scalable Model Management and Serving with Velox](#The-Missing-Piece-in-Complex-Analytics-Low-Latency-Scalable-Model-Management-and-Serving-with-Velox)

### Model and Lifecycle Management
- [ModelDB](#ModelDB)
- [ModelHub](#ModelHub)
- provDB 
- CometML
- [On Challenges in Machine Learning Model Management](#On-Challenges-in-Machine-Learning-Model-Management)
- [The Missing Piece in Complex Analytics: Low Latency, Scalable Model Management and Serving with Velox](#The-Missing-Piece-in-Complex-Analytics-Low-Latency-Scalable-Model-Management-and-Serving-with-Velox)
- [Scalable Deep Learning on Distributed Infrastructures](#Scalable-Deep-Learning-on-Distributed-Infrastructures)

### Recoverability/Reproducibility
- [dtoolAI: Reproducibility for Deep Learning](#dtoolAI-Reproducibility-for-Deep-Learning)

### Computational Graph (Formats) 
- [ONNX](#ONNX)

### Storage, Version, and Compression
#### models
- [ModelDB](#ModelDB)
- [ModelHub](#ModelHub)
- [Keystone ML](#Keystone-ML)
- Model Selection Management Systems: The Next Frontier of Advanced Analytics
- openML
- W3C Ml Schema
#### intermediates
- [MISTIQUE: A System to Store and Query Model Intermediates for Model Diagnosis](#MISTIQUE-A-System-to-Store-and-Query-Model-Intermediates-for-Model-Diagnosis)
#### data
- floats
    - Pstore: an efficient storage framework for managing scientific data
    - Isobar preconditioner for effective and high throughput lossless data compression
- datasets
    - Principles of dataset versioning: Exploring the recreation/storage tradeoff
    - DataHub
    - Decibel
- versioning of relational data
    - Orpheus DB
- multi dimensional arrays
    - TitleDB
    - SciDB
#### code
- git
#### storage and compression techniques/formats
- [Deep Compression](#Deep-Compression)
- Martin Kleppmann: Designing data-intensive applications (Chapter: Encoding and Evolution)
- delta encoding and compression
- python: pickle (used in PyTorch)
- zip (used in PyTorch)
- HD5 (used in TF)
- Huffman Encoding
        

### Tradeoffs (e.g. storage vs. runtime)
- [MISTIQUE: A System to Store and Query Model Intermediates for Model Diagnosis](#MISTIQUE-A-System-to-Store-and-Query-Model-Intermediates-for-Model-Diagnosis)
- Optimizing Machine Learning Workloads in Collaborative Environments


### Scientific workflow management
- [VisTrails](#VisTrails)
- Kepler
- Taverna

### Software engineering viewpoint
- A TensorFlow-Based Production-Scale Machine Learning Platform
- Hidden technical debt in machine learning systems
- Versioning for end-to-end machine learning pipelines
- Productionizing Machine Learning Pipelines at Scale.

### PyTorch links
- [tochvision models](https://pytorch.org/docs/stable/torchvision/models.html) 
- [Saving and loading models](https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference)
- [Reproducability](https://pytorch.org/docs/stable/notes/randomness.html)

## Reviewed (and relevant)

This is an overview file of related work in the domain of ML Model Managemnt

### MISTIQUE: A System to Store and Query Model Intermediates for Model Diagnosis
- MISTIQUE: **M**odel **I**ntermediate **ST**ore and **QU**ery **E**ngine
- not really *Model Management*, but *Model Intermediate Management*
- Goal: Efficiently capture, store and query **model intermediates** for diagnosis
- **interesting**: techniques to reduce storage footprint, cost models
- [notes](./mistique/README.md)

### Automatically Tracking Metadata and Provenance of Machine Learning Experiments
- **mainly interesting: presented [schema](https://github.com/awslabs/ml-experiments-schema)** 
- lightweight system to extract, store and manage metadata and prove- nance information of common artifacts in machine
 learning (ML) experiments
- tracking: datasets, models, predictions, evaluations and training runs
- [notes](./tracking-meta/README.md)

### On Challenges in Machine Learning Model Management
- **Model Management**: training, maintenance, deployment, monitoring, organization and documentation of machine 
learning (ML) models
- discuss a selection of ML use cases
- **overview of conceptual, engineering, and data-processing related challenges**
- point out future research directions
- [notes](./challanges/README.md)

### ModelDB
- Git like model version control based on research at MIT
- [website](https://www.verta.ai)
- [notes](./modelDB/README.md)

### Scalable Deep Learning on Distributed Infrastructures
- good overview on how to train a model in a distributed setting 
- especially interesting: 
    - describes data parallel training
    - section on Model (Data) Management, ref to ModelDB and ModelHUb
- [pfd](./dystdl/Scalable-Deep-Learning-on-Distributed-Infrastructures-Challenges-Techniques-and-Tools.pdf),
 so far no notes
 
### The Missing Piece in Complex Analytics: Low Latency, Scalable Model Management and Serving with Velox
- system for performing machine learning model serving and model maintenance at scale
- efficiently cache and replicate models across a cluster
- train models offline and online (might be relevant in our setting, further review needed)
- [pfd](./missing-piece/missing-piece.pdf), so far no extensive notes

### Deep Compression
- technique to compress CNNs with **no change** in prediction accuracy
    - maybe interesting for not to strict implementation of recoverability
- consists of 3 steps: pruning, enforce weight sharing, Huffman Encoding
- Drawbacks
    - results probably not exactly the same
    - to get to compressed version retraining required  
- [notes](./deep-compression/README.md)

### dtoolAI: Reproducibility for Deep Learning
- discusses Reproducibility, Replicability, and Repeatability in the context of DL
- implements dtoolAI - software to track metadata for datasets and for training with PyTorch 
- [notes](./dtoolai/README.md)

### reproduce vs replicate discussion
- we use the terms as follows: 
    - **reproduce**: same data+same methods=same results
    - **replicate**: new data and/or new methods in an independent study=same findings
- The document [Terminologies for Reproducible Research](./rep-vs-rec/Terminologies-for-Reproducible-Research.pdf)
discusses different, partly contradictorily definitions for the terms *reproduce* and *replicate* and defines three 
Categories. That the different of the term is not the same for all scientific work can also be seen by the fact that the
 ACM swapped the definitions for *reproduce* and *replicate* (see [ACM: Artifact Review and Badging](https://www.acm.org/publications/policies/artifact-review-and-badging-current)
- the definition of the terms how we use them is explained in a [separate readme](./rep-vs-rec/README.md)

### ONNX
- Open Neural Network Exchange (ONNX): [official website](https://onnx.ai/about.html)
- provides an open source format for AI models, both deep learning and traditional ML
- **current focus on the capabilities needed for inferencing (scoring)**
- [further notes and resources](./onnx/README.md)

### ModelHub
- data and lifecycle management system for deep learning
- design model versioning system similar to git 
- a read-optimized **parameter archival storage system (PAS) that minimizes storage footprint**
- develop efficient algorithms for archiving versioned models using deltas
- [notes](./modelHub))

## Reviewed (not too relevant)

### Online Model Management via Temporally Biased Sampling
- retraining of ML models in presence of evolving data
- present temporal biased sampling schemes
- result: increase ML accuracy and robustness with respect to evolving data
- [pdf](./online-mm/online-mm.pdf)

### VisTrails
- [website](https://www.vistrails.org/index.php/Main_Page)
- open-source scientific workflow and provenance management system that supports data exploration and visualization
- last version 05/2016, not maintained anymore  

### Keystone ML
- [Sparks, Evan R., et al. "Keystoneml: Optimizing pipelines for large-scale advanced analytics." 2017 IEEE 33rd 
international conference on data engineering (ICDE). IEEE, 2017.](https://arxiv.org/pdf/1610.09451)
- interesting aspect for us: "Allows users to specify end-to-end ML applications in a single system using high level 
logical operators"
- but operators are way too high-level for our use case 

### ML-Schema
- defining schema for ML experiments and compare the terms with other standards
- can be used to get inspiration what to think about, but schema not applicable in our setting, too highlevel
- last updated 2016
- [pdf](https://arxiv.org/pdf/1807.05351.pdf)
- [latest documentation](http://htmlpreview.github.io/?https://github.com/ML-Schema/documentation/blob/gh-pages/ML%20Schema.html)


## To Review TODO



- https://neptune.ai
- An Intermediate Representation for Optimizing Machine Learning Pipelines
- noWorkflow: a Tool for Collecting, Analyzing, and Managing Provenance from Python Scripts
- Automated Management of Deep Learning Experiments
- Model Selection Management Systems: The Next Frontier of Advanced Analytics
- Rondo: A Programming Platform for Generic Model Management
- Joaquin Vanschoren, Jan N Van Rijn, Bernd Bischl, and Luis Torgo. OpenML: networked science in machine learning. SIGKDD, 15(2):49–60, 2014
- Machine Learning Schema Community Group. W3c machine learning schema, 2017.
- Deep learning model management for coronary heart disease early warning research

- [Deployment and Model Management](https://link.springer.com/chapter/10.1007/978-3-030-45574-3_10)
- [A Programming System for Model Compression](http://learningsys.org/neurips19/assets/papers/16_CameraReadySubmission_WORKSHOP_VERSION_NeurIPS_2019.pdf)
- [How to put machine learning models into production](https://stackoverflow.blog/2020/10/12/how-to-put-machine-learning-models-into-production/?utm_source=Iterable&utm_medium=email&utm_campaign=the_overflow_newsletter)
- [TF - ML Metadata](https://www.tensorflow.org/tfx/guide/mlmd)
- [TF - ML Metadata - get started](https://github.com/google/ml-metadata/blob/master/g3doc/get_started.md)
- [Data Version Control](https://github.com/iterative/dvc)


## Related Research Directions 
from talk by [Manasi Vartak at DEEM](http://deem-workshop.org/videos/2020/7_vartak.mp4)
- Data Versioning
    - version control data similar to code 
    - existing work:
     [OrpheusDB](https://dl.acm.org/doi/abs/10.1145/3035918.3058744?casa_token=aHTGdV87tw4AAAAA:RYB2lh00gt7W3IZxoS4xSjXnljA-6HfAzX7qqhGkBvyLyT863fTm83PoGyGjXIZWRs4QrO0eApg), 
     [Principles of Dataset versioning](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5526644/),
     [DataHub](https://arxiv.org/abs/1409.0798)
- Data Linage
    - How was a dataset produced
    - existing work:
        - [Sub-zero](https://dspace.mit.edu/handle/1721.1/90854)
        - [ProvDB](http://sites.computer.org/debull/A18dec/p26.pdf)
- Model Debugging 
    - what data is the model predicting correctly vs. incorrectly, why?
    - existing work:
        - [MISTIQUE](./mistique/README.md)
        - Model Assertion (Stanford)
        - Model Diagnosis (UCB)  
- Model & Data Monitoring 
    - is the live model working similar to the offline model 
- Human-in-the-Loop ML
    - improving a model in real-time by annotating, re-training & testing 
    - Interactive Machine Learning