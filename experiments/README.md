# Experiments

## Datasets

### Imagenet Data 
- TODO describe the data 
- TODO link to Jupyter notebook

## Models  

## Pretrained Models: Used Data
We refer to the results and models listed in the 
[official PyTorch documentation](https://pytorch.org/docs/stable/torchvision/models.html) (last accessed, 01.12.2020)

Essential questions to answer for pre-trained models (AlexNet, VGG-19, ResNet18, ResNet50, Resnet152):
- What data was used to pre-train the models? (short answer: **data of ImageNet challenge 2012**)
- Was the validation dataset used to train the models? (short answer: **NO** )

What data was exactly used to train the models can not be answered by the information given in the documentation. 
Nevertheless, we can make some qualified guesses. 

What data was used to train the model? 
- for all models, the documentation says pretrained on ImageNet
- furthermore, the provided 
[dataloader uses the ImageNet data from 2012](https://github.com/pytorch/vision/blob/6e7ed49a93a1b0d47cef7722ea2c2f525dcb8795/torchvision/datasets/imagenet.py#L11-L15)
    - thus we can be pretty sure that the data used to pre-train the models is the **data of the ImageNet challenge 2012**. 
    The data can be downloaded [here](http://image-net.org/challenges/LSVRC/2012/downloads.php#images)

Was the validation set used to train the models?
- good overview of common definitions of train, test, and validation set can be found [here](https://machinelearningmastery.com/difference-test-validation-datasets/)
- they give the following definitions: 
    - Training Dataset: The sample of data used to fit the model.
    - Validation Dataset: The sample of data used to provide an unbiased evaluation of a model fit on the training dataset while tuning model hyperparameters. The evaluation becomes more biased as skill on the validation dataset is incorporated into the model configuration.
    - Test Dataset: The sample of data used to provide an unbiased evaluation of a final model fit on the training dataset.
- following these definitions, the validation dataset shouldn't be used to train the model, but it is also said: 
    - *the final model could be fit on the aggregate of the training and validation datasets*
- taking a look at the [GitHub issue](https://github.com/pytorch/vision/issues/2469) asking for the code that was used
 to generate the pre-trained models, we can find out that:
    - the validation split is used to generate the *dataset test* ([code](https://github.com/pytorch/vision/blob/6e7ed49a93a1b0d47cef7722ea2c2f525dcb8795/references/classification/train.py#L110-L138))
    - this data is **only** used to generate the *data_loader_test* ([code](https://github.com/pytorch/vision/blob/6e7ed49a93a1b0d47cef7722ea2c2f525dcb8795/references/classification/train.py#L164))
    - the *data_loader_test* is **only** used in the *evaluate* method ([code](https://github.com/pytorch/vision/blob/6e7ed49a93a1b0d47cef7722ea2c2f525dcb8795/references/classification/train.py#L48-L71))
    - the *evaluate* method calculates no gradients and also makes no use of the optimizer
    - thus **the validation set was not used to pretrain the models we use** 




