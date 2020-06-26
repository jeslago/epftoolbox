.. _metrics:

==================
Accuracy Metrics
==================
This module provides an easy-to-use interface to the most common and most suitable accuracy metrics in the context of day-ahead prices:

.. toctree::
   :maxdepth: 1
   
   metrics/mae
   metrics/rmse
   metrics/mape
   metrics/smape
   metrics/mase
   metrics/rmae

In addition, it also includes an implementation of the standard naive forecasts in electricity price forecasting. These forecasts are used to compute the :ref:`MASE <mase>` and :ref:`rMAE <rmae>` metrics:

.. toctree::
   :maxdepth: 1
   
   metrics/naive

**Standard Metrics**

In the field of electricity price forecasting, two of the most widely used accuracy metrics are the :ref:`mean absolute error (MAE) <mae>` and the :ref:`root mean square error (RMSE) <rmse>`:

.. math::

    \begin{align}
        \mathrm{MAE} &= \frac{1}{N}\sum_{k=1}^{N}|p_k-\hat{p}_k|,\\
        \mathrm{RMSE} &= \sqrt{\frac{1}{N}\sum_{k=1}^{N}(p_k-\hat{p}_k)^2},\\
    \end{align}

where :math:`p_k` and :math:`\hat{p}_k` respectively represent the real and forecasted prices at time step :math:`k`. 

Despite their popularity, the :ref:`MAE <mae>` and :ref:`RMSE <rmse>` are not always very informative as absolute errors are hard to compare between different datasets. In addition, :ref:`RMSE <rmse>` has the extra disadvantage of not representing accurately the underlying problem (electricity costs often depend linearly on prices but :ref:`RMSE <rmse>` is based on squared errors). 

Another standard metric is the :ref:`mean absolute percentage error (MAPE) <mape>`:

.. math::

    \begin{equation}
        \mathrm{MAPE} = \frac{1}{N}\sum_{k=1}^{N}\frac{|p_k-\hat{p}_k|}{|p_k|}.
    \end{equation}

While it provides a relative error metric that would grant comparison between datasets, its values become very large with prices close to zero (regardless of the actual absolute errors) and is also not very informative. 

Another propular metric is the :ref:`symmetric mean absolute percentage error (sMAPE) <smape>`:

.. math::

    \begin{equation}
        \mathrm{sMAPE} = \frac{1}{N}\sum_{k=1}^{N}2\frac{|p_k-\hat{p}_k|}{|p_k| + |\hat{p}_k|},\\       
    \end{equation}

Although the :ref:`sMAPE <smape>` solves some of these issues, it has a statistical distribution with undefined mean and infinite variance.

**MASE**

Arguably better metrics are those based on scaled errors, where a scaled error is simply the :ref:`MAE <mae>` scaled by the in-sample :ref:`MAE <mae>` of a naive forecast. A scaled error has the nice interpretation of being lower/larger than one if it is better/worse than the average naive forecast evaluated in-sample. 
A metric based on this concept is the :ref:`mean absolute scaled error (MASE) <mase>`, and in the context of one-step ahead forecasting is defined as:

.. math::

    \begin{equation}
        \mathrm{MASE} = \frac{1}{N}\sum_{k=1}^{N}\frac{|p_k-\hat{p}_k|}{\frac{1}{n-1}\sum_{i=2}^{n} |p^\mathrm{in}_i - p^\mathrm{in}_{i-1} |},
    \end{equation}

    
where :math:`p^\mathrm{in}_i` is the :math:`i^\mathrm{th}` price in the in-sample dataset and :math:`n` the size of the in-sample dataset.

**rMAE**

While scaled errors do indeed solve the issues of more traditional metrics, they have other associated problems that make them not unsuitable in the context of EPF:

    1. As :ref:`MASE <mase>` depends on the in-sample dataset, forecasting methods with different calibration windows  consider different in-sample datasets. Hence, the :ref:`MASE <mase>` of each model is based on a different scaling factor and comparisons between models cannot be drawn.
    2. Drawing comparisons across different time series is problematic as electricity prices are not stationary. For example, an in-sample dataset with spikes and an out-of-sample dataset without spikes will lead to a smaller :ref:`MASE <mase>` than if we consider the same market but with the in-sample/out-sample datasets reversed.

To solve these issues, an arguably better metric is the :ref:`relative MAE (rMAE) <rmae>` . Similar to :ref:`MASE <mase>`, :ref:`rMAE <rmae>` normalizes the :ref:`MAE <mae>` by the :ref:`MAE <mae>` of a naive forecast. However, instead of considering the in-sample dataset, the naive forecast is built based on the out-of-sample dataset. In the context of one-step ahead forecasting is defined as:

.. math::

    \begin{equation}
        \mathrm{rMAE} = \frac{1}{N}\sum_{k=1}^{N}\frac{|p_k-\hat{p}_k|}{\frac{1}{N-1}\sum_{i=2}^{N} |p_i - p_{i-1} |}.
    \end{equation}