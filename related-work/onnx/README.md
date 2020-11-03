# ONNX

- Open Neural Network Exchange (ONNX): [official website](https://onnx.ai/about.html)
- provides an open source format for AI models, both deep learning and traditional ML
- defines an extensible computation graph model, as well as definitions of built-in operators and standard data types
- **current focus on the capabilities needed for inferencing (scoring)**
    - can also be seen by the fact that most tutorials are about export, but not about training/import

## Further Resources
- the best for the documentation and tutorials is their [GitHub Repo](https://github.com/onnx/onnx)
- interesting tutorial: [Convert a PyTorch model to Tensorflow using ONNX](https://github.com/onnx/tutorials/blob/master/tutorials/PytorchTensorflowMnist.ipynb)
- [Portability between deep learning frameworks – with ONNX](https://blog.codecentric.de/en/2019/08/portability-deep-learning-frameworks-onnx/)
    - In TensorFlow and Caffe2 we are using a static graph to run computations. In PyTorch we are using a dynamic graph.
     The choose of the computation model can lead to some differences in programming and runtime. However, this is not 
     an issue for the ONNX standard. 
    - Limitations
        - if we have any custom/not supported layers operations we have to implement it for ONNX by ourselfs, 
        this can be time-consuming and laborious
        - we also have to double-check that the used (and supported) operations and functions are implemented in the 
        backends for the export and import.


### ONNX Runtime
- ONNX Runtime is a performance-focused engine for ONNX models, which inferences efficiently across multiple platforms and hardware
- [ONNX Runtime: a one-stop shop for machine learning inferencing](https://cloudblogs.microsoft.com/opensource/2019/05/22/onnx-runtime-machine-learning-inferencing-0-4-release/)



### Pytorch
- [Exporting a model from pytorch to onnx and running it using onnx runtime](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)
- the import of ONNX modles into PyTorch is not implemented yet, it is a [open feature request](https://github.com/pytorch/pytorch/issues/21683)
 since June 2019  
- [Torch.ONNX](https://pytorch.org/docs/master/onnx.html)
    - resulting alexnet.onnx is a binary protobuf file which contains both the network structure and parameters of the model you exported
    - trace based vs. script based
        - trace-based: 
            - run model one in prediction mode
            - export operators actually used during the run 
            - output only valid for a specific input size 
            (*comment*: have seen in Tutorial that at least to some extend variable)
            - loops will be unrolled, if conditions only the path of the run 
            - **recommend examining the model trace and making sure the traced operators look reasonable**
        - script-based:
            - exports into ScriptModule which is core data structure in TorchScript 
            - TorchScript is subset of Python, that creates serializable and optimizable models from PyTorch code.
        - mixing tracing and scripting is possible/allowed
    - **Limitations**
        - Only tuples, lists and Variables are supported as JIT inputs/outputs. Dictionaries and strings are also 
        accepted but their usage is not recommended
        - PyTorch and ONNX backends(Caffe2, ONNX Runtime, etc) often have implementations of operators with some numeric
         differences. [...] they **can also cause major divergences in behavior**
    - supported operators are listed [here](https://pytorch.org/docs/master/onnx.html#supported-operators)
        - the operator set is sufficient to export e.g.: ResNet, VGG, AlexNet, ...
        - *comment*: nothing said about possibility of training  
    - Training
        - setting a parameter in the exporting routine exports the model in a 'training friendly' way
        - it avoids certain model optimizations which might interfere with model parameter training
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    












