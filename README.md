# Unsupervised Learning For Auditory Data
This is the code corresponding to the experiments conducted for the project that use the work "Unsupervised Scalable Representation Learning for Multivariate Time Series" (Jean-Yves Franceschi, Aymeric Dieuleveut and Martin Jaggi) [[NeurIPS]](https://papers.nips.cc/paper/8713-unsupervised-scalable-representation-learning-for-multivariate-time-series) [[arXiv]](https://arxiv.org/abs/1901.10738) [[HAL]](https://hal.archives-ouvertes.fr/hal-01998101) for auditory data (speech data)

## Requirements

Experiments were done with the following package versions for Python 3.6:
 - Numpy (`numpy`);
 - Matplotlib (`matplotlib`);
 - Pandas (`pandas`);
 - PyTorch (`torch`);
 - Scikit-learn (`sklearn`);
 - Scipy (`scipy`)

This code should execute correctly with updated versions of these packages.

## Datasets

In this Project I used the LibriSpeech dataset to train the encoder. I tested the model on the test set of LibriSpeech and WSJ dataset.

The files `create_libri_set.py` `create_wsj_set.py` create the csv file that `libri.py` file uses

## Files

### Core

 - `losses` folder: implements the triplet loss in the cases of a training set
   with all time series of the same length, and a training set with time series
   of unequal lengths;
 - `networks` folder: implements encoder and its building blocks (dilated
   convolutions, causal CNN);
 - `scikit_wrappers.py` file: implements classes inheriting Scikit-learn
   classifiers that wrap an encoder and a SVM classifier.
 - `utils.py` file: implements custom PyTorch datasets;
 - `default_hyperparameters.json` file: example of a JSON file containing the
   hyperparameters of a pair (encoder, classifier).

### Report
There is a report elaborating the project in more detalis, named `Unsupervised learning project.pdf`

## Usage

### Training and testing

First, Set the hyperparameter you want in the default_hyperparameters.json file.

Train a model on the LibriSpeech dataset using this command:

`python libri.py --save_path /path/to/save/models`


In order test the model from trained model located in save_path path, run this command:

`python libri.py --save_path /path/to/save/models --load`




