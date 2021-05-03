# Knowledge Enhanced Neural Network

## Experiments on the Citeseer Dataset

This repository contains the experiments made with KENN on the <a href="https://linqs.soe.ucsc.edu/data" title="dataset">Citeseer Dataset</a>.

The original paper about KENN can be found <a href="https://link.springer.com/chapter/10.1007/978-3-030-29908-8_43">here</a>.

If you use this software for academic research, please, cite our work using the following BibTeX:

```
@InProceedings{10.1007/978-3-030-29908-8_43,
author="Daniele, Alessandro
and Serafini, Luciano",
editor="Nayak, Abhaya C.
and Sharma, Alok",
title="Knowledge Enhanced Neural Networks",
booktitle="PRICAI 2019: Trends in Artificial Intelligence",
year="2019",
publisher="Springer International Publishing",
address="Cham",
pages="542--554",
isbn="978-3-030-29908-8"
}
```

---

## Overview

KENN takes in input the predictions from a base NN and modifies them by exploiting logical knowledge given in input by the user in the form of universally quantified FOL clauses. It does so by adding a new final layer, called **Knowledge Enhancer (KE)**, to the existing neural network. The KE changes the orginal predictions of the standard neural network enforcing the satisfaction of the knowledge. Additionally, it contains **clause weights**, learnable parameters which represent the strength of each clause.

The Citeseer Dataset consists of 3312 scientific publications classified into one of six classes. The citation network consists of 4732 links. Each publication in the dataset is described by a 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary. The dictionary consists of 3703 unique words. The task is to correctly classify each scientific publication, given in input the features for each sample and the relational information provided by the citation network.

## The Prior Knowledge

For this experiment, the prior knowledge codifies the idea that papers cite works that are related to them (i.e. the topic of a paper is often the same of the paper it cites). For this reason the clause:

<div style="display:flex; justify-content:center; align-content:center; background:white; border-radius: 10px; width: 50vw; margin-right:auto; margin-left:auto; padding-top:0.5rem"><a href="https://www.codecogs.com/eqnedit.php?latex=\forall&space;x&space;\forall&space;y&space;T(x)&space;\land&space;Cite(x,y)&space;\rightarrow&space;T(y)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\forall&space;x&space;\forall&space;y&space;T(x)&space;\land&space;Cite(x,y)&space;\rightarrow&space;T(y)" title="\forall x \forall y T(x) \land Cite(x,y) \rightarrow T(y)" /></a></div>

was instantiated multiple times by substituting the topic _T_ with all the six classes.

## Learning Paradigms

KENN was trained following two different learning paradigms, which differ in the way the relational data from the citation network is used:

- The **Inductive** paradigm considers only the edges (x,y) such that both x and y are in the Training Set;
- The **Transductive** paradigm considers all the edges (x,y) in the citation network regardless of the Traning Set.

---

## How to Reproduce the Experiments

To replicate the experiments follow these steps:

1. Install the required python packages by running

```
pip install -r requirements.txt
```

2. Set the desired parameters inside `settings.py`. The Random Seed for our experiments was set to 0 but can be set to any other integer number.
3. To perform the tests, just run:

```
python all_tests.py [parameters]
```

```
  -h, --help  show this help message and exit
  -n N        The total number of runs for each learning paradigm;
  -ind        To perform the inductive training;
  -tra        To perform the transductive training;
  -gr         To perform the greedy training;
  -e2e        To perform the end to end training;
```

To run all the experiments using our settings, run:

```
python all_tests.py -n 500 -ind -tra -e2e
```

Results will be saved inside `./results/e2e`.

4. Inspect the results by running the `final_results.ipynb` notebook.

---

## License

Copyright (c) 2021, Daniele Alessandro, Mazzieri Riccardo, Serafini Luciano.

Licensed under the BSD 3-Clause License.
