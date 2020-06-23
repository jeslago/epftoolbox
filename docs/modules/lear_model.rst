.. _learref:

==================
LEAR
==================

The LEAR model is a parameter-rich ARX model estimated using the LASSO as an implicit feature selection that was originally proposed by `Uniejewski (2016) <https://doi.org/10.3390/en9080621>`_. It has been used in multiple studies and it has often shown state-of-the-art results in electricity price forecasting, e.g. see `Uniejewski (2016) <https://doi.org/10.3390/en9080621>`_ or `Lago (2018) <https://doi.org/10.1016/j.apenergy.2018.02.069>`_.

The LEAR model is provided in the library as a sigle :py:class:`LEAR <epftoolbox.models.LEAR>` class. The class receives as parameter the calibration window of the method, and has three four main function: a function to recalibrate the model, a function to make predictions, a function to recalibrate and predict, and a function that can perform daily recalibration and prediction using pandas DataFrames.

Besides the LEAR class, the library also includes the :py:func:`evaluate_lear_in_test_dataset <epftoolbox.models.evaluate_lear_in_test_dataset>` function. This function can be used as a simplified interface to evaluate a pandas DataFrame by simply specific the dates of the training and test datasets.

The library also includes a couple of :ref:`learex0` to get users familiar with the syntax and capabilities of the model.

.. toctree::
   :maxdepth: 2
   
   LEAR <lear/LEAR>
   evaluate_lear_in_test_dataset <lear/evaluate_lear_in_test_dataset>
