Welcome to epftoolbox's documentation!
======================================

This is the documentation of the epftoolbox. A toolbox and benchmark library to drive the research 
on day-ahead electricity price forecasting. 

The library contrain three main components: 

* The :ref:`data management<dataman>` subpackage, which comprises a module for :ref:`processing data<datawrang>` and another module for :ref:`dataset extraction<dataexct>`.

* The :ref:`models<models>` subpackage, which provides two state of the art forecasting models for electricity price forecasting. The contains a module for the :ref:`learref` model and another module for the :ref:`dnnref` model.

* The :ref:`evaluation<eval>` subpackage, which includes a module for evaluating the performance of the models in terms of :ref:`accuracy metrics<metrics>`, and another module to compare the forecasts of the models via :ref:`statistical testing<statest>`.

Using the index on the navigation bar or the index below you can navigate through the different library components.

.. toctree::
   :maxdepth: 1

   modules/started
   modules/data
   modules/models
   modules/evaluation 
   modules/examples

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
