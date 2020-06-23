# epftoolbox

The epftoolbox is the first open-access library for driving research in electricity price forecasting. Its main goal is to make available a set of tools that ensure reproducibility and establish research standards in electricity price forecasting research.

The library has been developed as part of the following article:

    Jesus Lago, Grzegorz Marcjasz, Bart De Schutter, Rafał Weron. "Forecasting day-ahead 
    electricity prices: A review of state-of-the-art algorithms, best practices and an 
    open-access benchmark". *Renewable and Sustainable Energy Reviews* (2020). Under Review.

Website: [https://epftoolbox.readthedocs.io/en/latest/](https://epftoolbox.readthedocs.io/en/latest/)

## Getting started
Download the repository and navigate into the folder
```bash
$ git clone https://github.com/jeslago/epftoolbox.git
$ cd epftoolbox
```
Install using pip
```bash
$ pip install .
```
Navigate to the examples folder and check the existing examples to get you started. The examples include several applications of the two state-of-the art forecasting model: a deep neural net and the LEAR model.

## Documentation
The documentation can be found [here](https://epftoolbox.readthedocs.io/en/latest/). It provides an introduction to the library features and explains all functionalities in detail. Note that the documentation is still being built and some functionalities are still undocumented.

## Features
The library provides easy access to a set of tools and benchmarks that can be used to evaluate and compare new methods for electricity price forecasting.

### Forecasting models
The library includes two state-of-the-art forecasting models that can be automatically employed in any day-ahead market without the need of expert knowledge. At the moment, the library comprises two main models:
  * One based on a deep neural network
  * A second based on an autoregressive model with LASSO regulazariton (LEAR). 

### Evaluation metrics
Standard evaluation metrics for electricity price forecasting including:
* Multiple scalar metrics like MAE, sMAPE, or MASE.
* Two statistical tests (Diebold-Mariano and Giacomini-White) to evaluate statistical differents in forecasting performance.

### Day-ahead market datasets
Easy access to five datasets comprising 6 years of data each and representing five different day-ahead electricity markets: 
* The datasets represents the EPEX-BE, EPEX-FR, EPEX-DE, NordPool, and PJM markets. 
* Each dataset contains historical prices plus two time series representing exogenous inputs.

### Available forecasts
Readily available forecasts of the state-of-the-art methods so that researchers can evaluate new methods without re-estimating the models.


## Citation
If you use the epftoolbox in a scientific publication, we would appreciate citations to the following paper:

    Jesus Lago, Grzegorz Marcjasz, Bart De Schutter, Rafał Weron. "Forecasting day-ahead 
    electricity prices: A review of state-of-the-art algorithms, best practices and an 
    open-access benchmark". *Renewable and Sustainable Energy Reviews* (2020). Under Review.


Bibtex entry::

    @article{epftoolbox,
     title={Forecasting day-ahead electricity prices: {A} review of state-of-the-art 
     algorithms, best practices and an open-access benchmark},
     author={Jesus Lago and Grzegorz Marcjasz and Bart De Schutter and Rafał Weron},
     journal={Renewable and Sustainable Energy Reviews},
     year={2020 (Under review)}
    }