# General notes for pytorch

## Links
- [saving and loading models](https://pytorch.org/tutorials/beginner/saving_loading_models.html?highlight=eval)

## model.eval() vs model.train()
- model.eval(): puts the model in evaluation mode -> self.train(False)
- model.train(): puts the model in train mode -> self.train(True)
- switching between these modes is important for layers that behave different between train and evaluation mode

#### Layers where the switch is important (only examples) 
- **Dropout layer** ([code](https://pytorch.org/docs/stable/_modules/torch/nn/modules/dropout.html#Dropout))
    - during training: randomly zeros out elements of the input 
    - during evaluation: identity mapping
- **Batch Normalization** ([code](https://pytorch.org/docs/master/_modules/torch/nn/modules/batchnorm.html#BatchNorm1d))
    - during training: tracks computed mean and variance and update values via momentum
    - during evaluation: in evaluation mode these values are fixed

