.. _eval:

==================
Model evaluation
==================

This subpackage provides a set of tools to evaluate forecasts and forecasting models in terms of accuracy metrics and statistical tests. 

The first subset of tools, which is provided by the :ref:`accuracy metrics <metrics>` module, evaluate the errors of the predictions based on a single value. They analyze how far the predictions are from the mean or median of the real prices. Examples of metrics are the mean absolute percentage error (MAPE) or the relative mean absolute error (rMAE). 

The second subset of tools, which is provided by the :ref:`statistical test <statest>` module, allows comparison between models by analyzing whether the difference in accuracy in the forecasts of the models is statistically significant. Unlike accuracy metrics, statistical tests allow to infer whether the difference in accuracy does really exist and it is not simply due to random differences between the forecasts.


.. toctree::
   :maxdepth: 2
   
   metrics
   stat_test




