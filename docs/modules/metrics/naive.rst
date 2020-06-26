==================
Naive forecast
==================

To compute the :ref:`rMAE <rmae>` and the :ref:`MASE <mase>`, a naive forecast is employed. The naive forecast can be built by three methods:

1. Considering daily seasonality and assuming that the prices from one day to the other do not change.
2. Considering weekly seasonality and assuming that the prices from one week to the other do not change
3. Considering different seasonality dependening on the day of the week: daily seasonality for Tuesday to Friday and weekly seasonality  for Saturday to Monday.

.. autofunction:: epftoolbox.evaluation.naive_forecast