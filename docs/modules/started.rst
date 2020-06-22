==================
Getting started
==================

Installation
----------------------
The library can be easily installed using pip. First clone the library and navigate to the folder:

.. code:: bash

    git clone https://github.com/jeslago/epftoolbox.git
    cd epftoolbox

Then, simply install the library using pip:

.. code:: bash

    pip install .

Functionality
----------------------
The library has three distinct modules: the :ref:`data management<dataman>` module, the :ref:`models<models>` module, and the :ref:`evaluation<eval>` module. 

The :ref:`first module<dataman>` provides functionality to manage, process, and obtain data for electricity price forecasting. The module also provides access to data from five different day-ahead electricity markets: EPEX-BE, EPEX-FR, EPEX-DE, NordPool, and PJM markets. 

The :ref:`second module<models>` grants access to state-of-the-art forecasting methods for day-ahead electricity prices that require no expert knowledge and can be automatically employed. At the moment, the library includes two state-of-the-art models: the :ref:`learref` model and the :ref:`dnnref` model.

The :ref:`third module<eval>` provides with an easy-to-use interface for evaluating forecasts in electricity price forecasting. This module includes both scalar metrics like MAE or MASE as well as statistical tests to evaluate the statistical difference in forecasting performance.

Using the library
----------------------
To learn how to use the library, there are three possibilities:

1. Since the library is rather simply, an user can easily read the library documentation of the specific modules that are of interest. The documentation includes explanations of each module and how to use them.

2. As an alternative, there are a number of :ref:`examples<examples>` that illustrate the most relevant functionalities of the library.

3. Finally, the library and its functionalities are explained in detail in the following article:

    J. Lago, G. Marcjasz, B. De Schutter, R. Weron. "Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark". *Renewable and Sustainable Energy Reviews (2020)*. Under Review.

For the most comprehensive introduction to the library, the user should first read the article, then the documentation, and finally go through the available :ref:`examples<examples>`.