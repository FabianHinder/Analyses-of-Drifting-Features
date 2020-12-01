# Analysis of Drifting Feature 

This repository contains the implementation of the methods proposed in the paper [Analysis of Drifting Features](Paper.pdf) by Fabian Hinder, Jonathan Jakob and Barbara Hammer.

## Requirements

- Python >= 3.6
- Packages as listed in [REQUIREMENTS.txt](REQUIREMENTS.txt)

## Third party components

- [squamish/](https://github.com/lpfann/squamish/tree/master/) is taken from [GitHub](https://github.com/lpfann/squamish) and implements of squamish by [Pfannschmidt et al 2020](https://arxiv.org/abs/2004.00658).
- Electricity market dataset ([original source](http://www.inescporto.pt/~jgama/ales/ales_5.html), [normalized](http://moa.cms.waikato.ac.nz/datasets/)) dataset described by M. Harries and analysed by Gama 
- Poker-Hand ([original source](https://archive.ics.uci.edu/ml/datasets/Poker+Hand), [normalized](https://sourceforge.net/projects/moa-datastream/files/Datasets/Classification/poker-lsn.arff.zip/download/)) dataset by R. Cattral and F. Oppacher

## Installation

#### Squamish
The squamish folder contains the Relevance Bounds Method. First, you need to install poetry (https://python-poetry.org/).
Then, you can run 'poetry install' inside the squamish folder. If you use pycharm, mark the folder as 'sources root' afterwards.

#### Statistical DFA
The statistic_DFA file contains the statistical DFA method. To use it you need to install the python libraries fcit and causaldag.
You can do so with 'pip install fcit' and 'pip install causaldag'.

## Data sets

The data for the experiments is located in the data folder.

#### Real World data

The Electricity and Poker data sets are well known benchmarks, that are referenced in the paper.

#### Theoretical data

Our own data sets with known ground truth can be distinguished by their file names. 

- T: marks the time feature 
- C: marks the non-drifting features 
- F: marks the faithfully-drifting features 
- I: marks the drift-inducing features 

The features are arranged in the order T,C,F,I.

Additional data sets can be created using ground_truth.py

## Experiments

To recreate our experiments, simply run the experiments file.

## How to cite

You can cite the version on TODO
