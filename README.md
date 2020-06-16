# A benchmark and toolbox library for electricity price forecasting

An open-access benchmark and toolbox to help drive the research in electricity price 
forecasting. 

It provides access to a set of tools and benchmarks that can be used to evaluate and compare
new methods for electricity price forecasting. The main features of the library are:

* State-of-the-art forecasting models that can be automatically employed in
any day-ahead market without the need of expert knowledge. At the moment, the library comprises
two main models:
  * One based on a deep neural network
  * A second based on an autoregressive model with LASSO regulazariton. 

* Standard evaluation metrics for electricity price forecasting:
  * Multiple accuracy metrics to evaluate performance: MAE, MAPE, sMAPE, MASE, rMAE, RMSE, and rRMSE.
  * Two statistical tests (Diebold-Mariano and Giacomini-White) to evaluate statistical differents in forecasting performance.

* Five datasets comprising 6 years of data each and representing five different day-ahead electricity markets:
  * Markets: EPEX-BE, EPEX-FR, EPEX-DE, NordPool, and PJM
  * Data: each dataset contains historical prices plus two time series representing exogenous inputs, 

* Ready available forecasts of the state-of-the-art methods so that researchers can evaluate new methods 
without re-estiamting models.


----

To cite the library, please use the following reference:

Jesus Lago, Grzegorz Marcjasz, Bart De Schutter, Rafał Weron. "Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark". Renewable and Sustainable Energy Reviews (2020). Under Review.