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
- [related work](./related-work)
- elaborate forms of model comparison: MISTIQUE
- basic description of an experiment model that could be used for storing the pipelines: "Schema Paper"

**Other**
- further [notes and research questions](./notes)
	

        


 

