.. _models:

==================
Forecasting models
==================

The subpackage provides an easy interface to two state-of-the-art forecasting models in the field of
electricity price forecasting: the :ref:`LEAR` and the :ref:`DNN` models. 

For the :ref:`LEAR`, it provides an interface to perform estimation, daily recalibration,
and prediction. For :ref:`DNN` model, it provides an interface to perform estimation,
hyperparameter optimization, daily recalibration, and prediction.  

A detailed explanations of the models can be obtained in:

	J. Lago, G. Marcjasz, B. De Schutter, R. Weron. "Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark". *Renewable and Sustainable Energy Reviews (2020)*. Under Review.

.. _learref:

LEAR
----------------------

.. autoclass:: epftoolbox.models.LEAR
   :members:


.. _dnnref:

DNN
----------------------

.. autoclass:: epftoolbox.models.DNNModel
   :members:

.. autoclass:: epftoolbox.models.DNN
   :members:

.. autofunction:: epftoolbox.models.hyperparameter_optimizer