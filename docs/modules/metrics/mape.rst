==================
MAPE
==================
Another popular metric for electricity price forecasting is the mean absolute percentage error (MAPE):

.. math::

    \begin{equation}
        \mathrm{MAPE} = \frac{1}{N}\sum_{k=1}^{N}\frac{|p_k-\hat{p}_k|}{|p_k|}.
    \end{equation}

This metric computes the :ref:`MAE <mae>` between the predicted prices and the real prices and normalizes it by the absolute value of the real prices. 

While it provides a relative error metric that would grant comparison between datasets, its values become very large with prices close to zero (regardless of the actual absolute errors) and is also not very informative. 



epftoolbox.evaluation.MAPE
---------------------------

.. autofunction:: epftoolbox.evaluation.MAPE