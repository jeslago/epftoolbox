# epftoolbox


An open-access benchmark and toolbox to help drive the research in electricity price forecasting. 

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
We are still working on the documentation. You can start in the meantime by looking at the examples and reading the paper where the library was proposed:

Jesus Lago, Grzegorz Marcjasz, Bart De Schutter, Rafał Weron. "Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark". *Renewable and Sustainable Energy Reviews* (2020). Under Review.

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
