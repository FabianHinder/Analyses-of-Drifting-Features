# On the Fine-structure of Drifting Features

Experimental code of conference paper. [Paper](TODO) 

## Abstract

Feature selection is one of the most relevant preprocessing and analysis techniques in machine learning, allowing for increases in model performance and knowledge discovery. In online setups, both can be affected by concept drift, i.e., changes of the underlying distribution. Recently an adaption of classical feature relevance approaches to drift was introduced. While the method increases detection performance significantly, there is only little discussion on the explanatory aspects. In this work, we focus on understanding the structure of the ongoing drift by transferring the concept of strongly and weakly relevant features. We empirically evaluate our methodology using graphical models.

## Requirements

* Python 
* Numpy, SciPy, Pandas, Matplotlib
* scikit-learn
* BorutaPy

## Usage

To run the experiments, there are three stages 1. create the datasets (`--make`) which creates the datasets and stores them in a local directory, 2. splits the experimental setups in several chunks (`--setup #n`) for parallel processing on different devices, and 3. running the experiments (`--run_exp #n`) which runs the chunk as indicated by the command line attribute.

## Cite

Cite our Paper
```
TODO
```

## License

This code has a MIT license.

