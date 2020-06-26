====================
MASE
====================

Considering the errors of standard metrics described in the :ref:`introduction <metrics>`, metrics based on scaled errors, where a scaled error is simply the :ref:`MAE <mae>` scaled by the in-sample :ref:`MAE <mae>` of a :py:class:`naive forecast <epftoolbox.evaluation.naive_forecast>`, are arguably better. A scaled error has the nice interpretation of being lower/larger than one if it is better/worse than the average naive forecast evaluated in-sample. 

A metric based on this concept is the mean absolute scaled error (MASE), and in the context of one-step ahead forecasting is defined as:

.. math::

	\begin{equation}
		\mathrm{MASE} = \frac{1}{N}\sum_{k=1}^{N}\frac{|p_k-\hat{p}_k|}{\frac{1}{n-1}\sum_{i=2}^{n} |p^\mathrm{in}_i - p^\mathrm{in}_{i-1} |},
	\end{equation}

	
where :math:`p^\mathrm{in}_i` is the :math:`i^\mathrm{th}` price in the in-sample dataset and :math:`n` the size of the in-sample dataset. For seasonal time series, the MASE may be defined using the :ref:`MAE <mae>` of a seasonal naive model in the denominator:

.. math::

	\begin{equation}
	\mathrm{MASE}_{m} = \frac{1}{N}\sum_{k=1}^{N}\frac{|p_k-\hat{p}_k|}{\frac{1}{n-m}\sum_{i=m+1}^{n} |p^\mathrm{in}_i - p^\mathrm{in}_{i-m} |}
	\end{equation}

where :math:`m` represents the seasonal length  (in the case of day-ahead prices that could be either 24 or 168 representing the daily and weekly seasonalities). As an alternative, the :py:class:`naive forecast <epftoolbox.evaluation.naive_forecast>` can also be defined on the standard naive forecast for price forecasting (using daily seasonality for Tuesday to Friday and weekly seasonality for Saturday to Monday).

epftoolbox.evaluation.MASE
---------------------------

.. autofunction:: epftoolbox.evaluation.MASE