# Master Thesis

This is the corresponding repository to my master thesis with the title **Model Management in Distributed Environments**
.

## Abstract

Over the last few years, deep learning (DL) has revolutionized many domains by significantly outperforming previous
approaches and became essential for many software products. To guarantee reliable and consistent performance, models
that are used need not only to be adjusted, improved, and retrained but also documented, deployed, and monitored from a
central location. An essential part of this set of processes, referred to as model management (MM), is to save and
recover models; which ideally happens without loss of precision. Existing approaches in MM either compress the model
with loss of precision or focus on metadata and use a naive approach to saving a model. In this thesis we investigate
if, and by how much, we can outperform a baseline approach capable of naively saving and recovering models without loss
of precision, while focusing on metrics of storage consumption, time-to-save, and time-to-recover.

We develop a set of approaches consisting of a baseline approach *B* that saves complete model snapshots, a parameter
update approach *U_p′* that saves the updated model data, and a provenance approach *M_Prov* that saves the model’s
provenance instead of the model itself. In addition to these approaches, we also develop a probing tool to determine if
we can reproduce the inference and training of a given model across different machines. Evaluating all approaches in a
distributed environment on different model architectures, model datasets, and model relations, we show that *M_Prov*
outperforms the baseline by up to 70% and *U_p′* by up to 95.6% in terms of storage consumption. While both approaches
have the potential to achieve a similar or slightly shorter time-to-save, they come with a longer time-to-recover. We
find that if and by how much, a given approach outperforms a baseline strongly depends on factors such as the dataset
used, the model architecture, and how many model parameters stay fixed across different model versions.

## MMlib

- the *MMlib* developed as part of the thesis comprising all our approaches is *NOT* part of this repo
    - you can find it here: [mmlib](https://github.com/slin96/mmlib)

## Repo Structure

- thesis in PDF format
    - [thesis](./thesis.pdf)
- overview of related work
    - [related work](./related-work)
- the code for all experiments
    - [experiments](./experiments)
- description and creation of the datasets we used
    - [data](./data) 
  

