# Overview

![alt text](images/distributed-models.png "idea")

**Idea**
- managing many, slightly different models in a distributed setup

**Possible Use Case**
- models in cars, mobile phones, or industrial equipment (turbines, machines, etc.)

**Setting**
- the models are deployed centrally
- but might be updated locally based on new information
- The new model must be known / recoverable by the central server

**Baseline** 
- system that receives and stores all instances of individual models

**Possible Approach**
- local instances could send gradient updates 
- or just pipeline descriptions and raw data, if this is more efficient and deterministically computable

**References**
- elaborate forms of model comparison: MISTIQUE
- basic description of an experiment model that could be used for storing the pipelines: "Schema Paper"
	
## Possible research questions
- Best way to represent/save/transfer model? - Evaluate different strategies in different settings  
- What do we have to save to make a model recoverable (and how do we define recoverable)? 
- Develop a system architecture + data model to manage multiple different models in a distributed setting 
(Distributed Model Management / Distributed Model Versioning)
- Develop ways to (mathematically) model: retrain time, used bandwidth to transfer data + param-hash, parameters + pipeline, ...
    - define a tradeoff value(s) similar to e.g. 
        - MISTIQUE's storage vs. saving query time for calc. intermediate
        - opt ML workloads's storage vs. recompute time  
- Think about a partly leaderless setting, remove workload from 'master' in P2P fashion? 
    
## Things to think about in described setting  

### General 
- How does this setting differ to the setting of "data parallel training" of ML models
- what do we mean by certain terms e.g. recoverable 
    - just have a model that produces same results (only weights must be transferred)
    - or ability to retrain the model on central server/node (data must be transferred,
     maybe hash as rof weights to ensure same results)
- what assumptions are valid to make?
    - e.g.: every instance has at least:
        - xGB memory available
        - enough computation power to: train model, calc diffs between models, ...

### Model (update)
- How do we represent the model?
- What needs to be deployed to the cars? 
- What must be sent back to know/recover the model centrally?
- Do we have specialized models form beginning (e.g. slightly different for car model)
- What operations are possible on the target device to get to the final model (relevant if we e.g. only send data +
 (un-/pre-)trained model) 
- How do we calculate the diff between models/gradients?
- Is sending data plus model pipeline more efficient, thank model + weights? 
- Merkle tree for efficient diff of pipelines, weights, ...?

### Data
- How does the data, seen by different instances, varies
- is the data collected at single instances 'biased'
    - daytime - e.g. drives only during night
    - weather - always rain/sun
    - region - only specific country, region, highway vs city
    - ...
- To what extent do we have to store metadata
    - e.g. probably don't need score for every epoch
    
### Instances (e.g. Cars) 
- is a model trained on a specific instance feasible for other cars?
- to what extent are the cars different
    - hardware
    - specific of car influence model performance, training, ...
    - ...



        


 

