# HaloEGNN

HaloEGNN is a library that models dark matter halos with Equivariant Graph Neural Network.

A key yet unresolved question in modern-day astronomy is how galaxies formed and evolved under the paradigm of the ΛCDM model. A critical limiting factor lies in the lack of robust tools to describe the merger history through a statistical model. In this work, we employ a generative graph network, E(n) Equivariant Graph Normalizing Flows Model. We demonstrate that, by treating the progenitors as a graph, our model robustly recovers their distributions, including their masses, merging redshifts and pairwise distances at redshift z=2 conditioned on their z=0 properties. The generative nature of the model enables other downstream tasks, including likelihood-free inference, detecting anomalies and identifying subtle correlations of progenitor features.

[Accepted to ICML 2022 Machine Learning for Astrophysics Workshop](https://arxiv.org/abs/2207.02786)


## Installation
minimal package requirement
```
pip install -r requirement
pip install -e .
```

## Usage
```python

```
