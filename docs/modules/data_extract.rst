.. _dataexct:

====================
Dataset extraction
====================
This module provides an easy-to-use interface to download data from multiple day-ahead electricity markets using the following `database <https://zenodo.org/records/4624805>`_. The module is built around the function :py:class:`read_data <epftoolbox.data.read_data>`, and it can be used to obtain the market data from the following periods and day-ahead electricity markets:

=============  ==========================
   Market       Period
=============  ==========================
Nord pool  		01.01.2013 – 24.12.2018  
PJM   		  	01.01.2013 – 24.12.2018  
EPEX-France     09.01.2011 – 31.12.2016
EPEX-Belgium    09.01.2011 – 31.12.2016
EPEX-Germany	09.01.2012 – 31.12.2017
=============  ==========================

Besides the data from these five markets, the module also provides an interface to read `csv` files from other markets and transform their data to match the naming requirements of the prediction models in the epftoolbox library.  In addition, it also implements an automatic training/testing split based on the testing period under study. 

.. autofunction:: epftoolbox.data.read_data
