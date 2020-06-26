==================
MAE
==================
In the field of electricity price forecasting, one of the most widely used metrics to measure the accuracy of point forecasts is the mean absolute error (MAE):

.. math::

    \begin{align}
        \mathrm{MAE} &= \frac{1}{N}\sum_{k=1}^{N}|p_k-\hat{p}_k|,\\
    \end{align}

This metric computes the average absolute error between the predicted prices and the real prices. Predictive models that minimize the MAE lead to predictions of the median of the prices. Despite its popularity, the :ref:`MAE <mae>` is not always very informative as absolute errors are hard to compare between different datasets.

epftoolbox.evaluation.MAE
---------------------------

.. autofunction:: epftoolbox.evaluation.MAE