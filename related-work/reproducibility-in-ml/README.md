# Reproducibility in Machine Learning

- The main source for the notes is
  [part1](https://suneeta-mall.github.io/2019/12/21/Reproducible-ml-research-n-industry.html) of the blog series

## Importance of reproducibility

- reaction to *repeatability crisis*, for example:
    - [ML Reproducibility Challenge 2020](https://paperswithcode.com/rc2020)
    - [NeurIPS: The Machine Learning Reproducibility Checklist](https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf)

- the articles list also some reasons why reproducibility is important
    - Understanding, Explaining, Debugging and Reverse Engineering
    - Correctness
    - Credibility (Lecun: "Good results are not enough, Making them easily reproducible also makes them credible")
    - Extensibility
    - Data harvesting

## Challenges in realizing reproducible ML

The blog gives a good overview of factors to consider when implementing reproducible ML. The author groups the factors
into five categories, we only name there most important onew here

- **Hardware**
    - here the most important factor is floating-point numbers
    - references given are:
        - [Presentation by Intel: Consistency of Floating Point Results](./reproducibility-in-ml/FloatingPoint_consistency.pdf)
            - gist: discussion of compiler options and the fact that some transformations (e.g. a+b+c=(a+b)+c=a+(b+c))
              are equivalent mathematically but not in finite precision arithmetic

        - [Wandering Precision](./reproducibility-in-ml/Wandering-Precision.pdf)
            - gist: modern hardware uses optimizations (e.g. SSE instructions) for floating-point arithmetic, this leads
              to the fact that the order of computation is not always the same which then leads to different result
- **Software**
    - not all software works in a way so that it is reproducible
    - for example the frameworks Tensorflow and Pytorch do not guarantee to 100% reproducible
    - also CUDA does not guarantee reproducibility for all their routines/in all configurations
- **Algorithm**
    - also the implemented algorithm can not reproducible because it makes use of randomization
    - examples in DL are:
        - Random data augmentation
        - Shuffle the dataset
        - Random wight initialization
        - Dropout Layers
  


