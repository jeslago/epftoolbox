.. _datawrang:

==================
Data wrangling
==================
This module is intended for transforming data into a format that can be read and processed by the prediction models of the epftoolbox library. At the moment, the module is limited to scaling operations. 

The module is composed of two components:

   - The :py:class:`DataScaler <epftoolbox.data.DataScaler>` class.
   - The :py:class:`scaling <epftoolbox.data.scaling>` function.

The class :py:class:`DataScaler <epftoolbox.data.DataScaler>` is the main block for performing scaling operations. The class is based on the syntax of the scalers defined in the `sklearn.preprocessing <https://scikit-learn.org/stable/modules/preprocessing.html>`_ module of the scikit-learn library. The class performs some of the standard scaling algorithms in the context of electricity price forecasting:


Besides the class, the module also provides a function :py:class:`scaling <epftoolbox.data.scaling>` to scale
a list of datasets but estimating the scaler using only one of the datasets. This function is useful when
scaling the training, validation, and test dataset. In this scenario, to have a realistic evaluation, one would ideally estimate the scaler using the training dataset and simply transform the other two.

------------

.. autoclass:: epftoolbox.data.DataScaler
   :members:

------------
.. autofunction:: epftoolbox.data.scaling

