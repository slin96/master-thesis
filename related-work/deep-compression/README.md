# Deep Compression

- technique to compress CNNs with **no change** in prediction accuracy
- maybe interesting for not to strict implementation of recoverability
- [pdf](./deep-compression.pdf)

### Drawbacks

- if the results of the network are exactly the same is not said
    - probably not, probably predicted probability for classes slightly shifted but still same class predicted
- to get to compressed version retraining required

## Model Compression

- consists of 3 steps: pruning, enforce weight sharing, Huffman Encoding

### pruning

- all connections with weights below a threshold removed
- retrain to learn final weights
- weights are stored using compressed sparse row (CSR) or compressed sparse column (CSC) format

### weight sharing

- reduce the number of effective weights by making connections share a weight
- k-means clustering to identify the shared weights for each layer -> clusters share a weight
- use one of presented techniques to chose fial weight for cluster e.g. just calc. mean value

### Huffman Coding

- used to finally compress the model
- optimal prefix code commonly used for lossless data compression
- works good if the symbols are not evenly distributed, which is the cas ein their experiments

