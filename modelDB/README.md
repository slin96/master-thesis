# ModelDB

## ModelDB 1.0: Paper [1] Manasi Vartak (2016, MIT) 
- **end-to-end system for the management of machine learning models**
- Model management is the problem of tracking, storing and indexing large numbers of machine learning models so they may subsequently be shared, queried and analyzed
- Model management provides support for: 
    - recapitulate insights
    - sensemaking
    - find trends and perform meta-analyses across models
    - search through models
    - collaboration
- managing models means managing multi stage pipelines -> manage also: 
    - metadata (parameters of pre-processing steps, hyperparameters for models etc.)
    - quality metrics
    - datasets
    
### Architecture
- client side 
    - native client libraries for:
        - spark.ml
        - scikit-learn
    - frontend: web-based visualization interface
        - review all models and pipelines
        - quickly understand and inspect pipelines
        - comparisons across pipelines and models 
- backend
    - defines key abstractions and brokers access to the storage layer
    - stores models and pipelines as a sequence of actions
    - storage layer
        - relational DB
        - custom engine to store and index models 
    - **Not described: Any details for Datamodel/Schema**

    
    

    
    
    
- Further work lead to product modelDB 2.0 [verta.ai](https://www.verta.ai)


## Sources
- [[1]](https://dl.acm.org/doi/abs/10.1145/2939502.2939516?casa_token=B1-fF_wNvdgAAAAA:pduTz2ZCbgbYHsmOQETTKTtb4QM6Z01VTm52j6sgiOTeU8J_W2kDDoBf06r0-wTctQV9o3ZSgYE) Vartak, Manasi, et al. "ModelDB: a system for machine learning model management." Proceedings of the Workshop on Human-In-the-Loop Data Analytics. 2016.
- [[2]](https://www.youtube.com/watch?v=U0lyF_lHngo) Vartak, Manasi: "Model Versioning Done Right: A ModelDB 2.0 Walkthrough"
- [[3]](https://github.com/VertaAI/modeldb) GitHub: VertaAI/modeldb 