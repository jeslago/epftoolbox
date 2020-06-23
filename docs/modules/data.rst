.. _dataman:

==================
Data management
==================

This subpackage provides an interface to extract data from different day-ahead electricity markets and a module to process the market data before its use in prediction models. 

The first functionality is provided by the :ref:`data extraction <dataexct>` module,  which provides automatic access to the data from five different day-ahead electricity markets as well as an easy-to-use interface to read data from other markets.

The second functionality is provided by the :ref:`data wrangling <datawrang>` module, which includes the most common scaling transformations in electricity price forecasting. 


.. toctree::
   :maxdepth: 1
   
   data_extract
   data_wrangling
   

.. For the :ref:`LEAR` model, the subpackage provides an interface to perform estimation, daily recalibration,
.. and prediction. For :ref:`DNN` model, it provides an interface to perform estimation,
.. hyperparameter optimization, daily recalibration, and prediction.  
