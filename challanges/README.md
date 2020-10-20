# On Challenges in Machine Learning Model Management

- The focus of the following notes is: Challenges in ML Model Management 
- Problems have their roots in 
    - complexity of ML systems
    - lack of declarative abstraction 
    - heterogeneity of the resulting codebases
- questions of model management encounter a growing interest in academic community  

## Conceptual Challenges

- Machine Learning Model Definition
    - difficult to define the actual model to manage
    - not only model parameters/weights are important but also preprocessing steps (e.g. feature transformation)
    - do we consider model as 'black-box'?
    - do we present model in form of comprised of operations with known semantics
    - complexity of many real-world ML applications make definition of model even more difficult

- Model Validation
    - every time when: data changes/new software dependencies/improved model -> performance must be re-validated
        - must ensure:
            - trained and evaluated using same training, test and validation set
            - same code for evaluating metrics
            
- Decision on Model Retraining (for now not of interest)
- Adversarial Settings (for now not of interest)

## Data Management Challenges

- Lack of a Declarative Abstraction for the whole ML Pipeline
    - for training of DNNs development of specialized systems 
    - preprocessing mostly: map-reduce like transformations
    - -> pipeline consists of different systems 
    - problems
        - complicates extraction of metadata
        - difficult replicability and automation of model selection
        
- Querying Model Metadata
    - many existing ML frameworks have not been designed to automatically expose their metadata
    
## Engineering Challenges
- Multi-Language Code Bases
- Heterogeneous Skill Level of Users
- Backward Compatibility of Trained Models
    - different degrees of backward compatibility, after e.g. a year
        - exact same results
        - similar results 
        - the model can still run
    - need to store at least 
        - data transformations 
        - implementation of algo. with configuration and dependencies
        
 
