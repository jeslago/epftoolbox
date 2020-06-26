======
rMAE
======

While scaled errors do indeed solve the issues of more traditional metrics, they have other associated problems that make them not unsuitable in the context of EPF:

    1. As :ref:`MASE <mase>` depends on the in-sample dataset, forecasting methods with different calibration windows will naturally have to consider different in-sample datasets. As a result, the :ref:`MASE <mase>` of each model will be based on a different scaling factor and comparisons between models cannot be drawn.
    2. The same argument applies to models with and without rolling windows. The latter will use a different in-sample dataset at every time point while the former will keep the in-sample dataset constant.
    3. In ensembles of models with different calibration windows, the :ref:`MASE <mase>` cannot be defined as the calibration window of the ensemble is undefined.
    4. Drawing comparisons across different time series is problematic as electricity prices are not stationary. For example, an in-sample dataset with spikes and an out-of-sample dataset without spikes will lead to a smaller :ref:`MASE <mase>` than if we consider the same market but with the in-sample/out-sample datasets reversed.

To solve these issues, an arguably better metric is the relative MAE (rMAE). Similar to :ref:`MASE <mase>`, rMAE normalizes the :ref:`MAE <mae>` by the :ref:`MAE <mae>` of a naive forecast. However, instead of considering the in-sample dataset, the naive forecast is built based on the out-of-sample dataset. In the context In the context of one-step ahead forecasting is defined as:

.. math::

    \begin{equation}
        \mathrm{rMAE} = \frac{1}{N}\sum_{k=1}^{N}\frac{|p_k-\hat{p}_k|}{\frac{1}{N-1}\sum_{i=2}^{N} |p_i - p_{i-1} |}.
    \end{equation}

For seasonal time series, the rMAE may be defined using the :ref:`MAE <mae>` of a seasonal naive model in the denominator:

.. math::

	\begin{equation}
	\mathrm{rMAE}_{m} = \frac{1}{N}\sum_{k=1}^{N}\frac{|p_k-\hat{p}_k|}{\frac{1}{N-m}\sum_{i=m+1}^{N} |p_i - p_{i-m} |}
	\end{equation}

where :math:`m` represents the seasonal length  (in the case of day-ahead prices that could be either 24 or 168 representing the daily and weekly seasonalities). As an alternative, the naive forecast can also be defined on the standard naive forecast for price forecasting (using daily seasonality for Tuesday to Friday and weekly seasonality for Saturday to Monday).

epftoolbox.evaluation.rMAE
---------------------------

.. autofunction:: epftoolbox.evaluation.rMAE