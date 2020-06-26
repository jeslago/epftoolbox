==================
sMAPE
==================
Another popular metric in the field of electricity price forecasting is the symmetric mean absolute percentage error (sMAPE):

.. math::

    \begin{equation}
        \mathrm{sMAPE} = \frac{1}{N}\sum_{k=1}^{N}2\frac{|p_k-\hat{p}_k|}{|p_k| + |\hat{p}_k|},\\       
    \end{equation} 

This metric computes the :ref:`MAE <mae>` between the predicted prices and the real prices and normalizes it by the average of the absolute value of both quantities. Note, that there are `multiple versions <https://robjhyndman.com/hyndsight/smape/>`_ of sMAPE and here we consider the most `sensible one <https://robjhyndman.com/hyndsight/smape/>`_.

Although the sMAPE  provides a metric based on relative errors that would grant comparison between datasets and even though it solves some of the issues of :ref:`MAE <mae>`, :ref:`RMSE <RMSE>`, and :ref:`MAPE <MAPE>` and, it has a statistical distribution with undefined mean and infinite variance.

epftoolbox.evaluation.sMAPE
----------------------------

.. autofunction:: epftoolbox.evaluation.sMAPE