# Related Work

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

## Reviewed (not too relevant)

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

## To Review TODO
- https://neptune.ai
- An Intermediate Representation for Optimizing Machine Learning Pipelines
- noWorkflow: a Tool for Collecting, Analyzing, and Managing Provenance from Python Scripts
- ModelHub: Deep Learning Lifecycle Management
- Automated Management of Deep Learning Experiments
- DEEM 2019: Workshop on Data Management for End-to-End Machine Learning
- Model Selection Management Systems: The Next Frontier of Advanced Analytics
- Rondo: A Programming Platform for Generic Model Management
- Joaquin Vanschoren, Jan N Van Rijn, Bernd Bischl, and Luis Torgo. OpenML: networked science in machine learning. SIGKDD, 15(2):49–60, 2014
- Machine Learning Schema Community Group. W3c machine learning schema, 2017.
- Hui Miao, Ang Li, Larry S Davis, and Amol Deshpande. Towards unified data and lifecycle management for deep learning. In ICDE, pages 571–582, 2017.
- Deep learning model management for coronary heart disease early warning research
- [Reproducibility for Deep Learning](https://www.sciencedirect.com/science/article/pii/S2666389920300933)
- [Deployment and Model Management](https://link.springer.com/chapter/10.1007/978-3-030-45574-3_10)
- [A Programming System for Model Compression](http://learningsys.org/neurips19/assets/papers/16_CameraReadySubmission_WORKSHOP_VERSION_NeurIPS_2019.pdf)

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
        


 

