.. _learex0:

==================
LEAR Examples
==================
This section contains two examples on how to use the :ref:`LEAR` model. The :ref:`first example<learex1>` provides an easy-to-use interface for evaluating the LEAR model in a given test dataset. The :ref:`second example<learex2>` provides more flexible interface to perform recalibration and daily forecasting with a LEAR model. 

.. _learex1:

1. Easy recalibration
------------------------
The first example provides an easy-to-use interface for evaluating the LEAR model in a given test dataset. While this example lacks flexibility, it grants an simple interface to evalute LEAR models
in different datasets.

.. literalinclude:: ../../examples/recalibrating_lear_simplified.py
  :language: python

.. _learex2:

2. Flexible recalibration
--------------------------
The second example provides more flexible interface to perform recalibration and daily
forecasting with a LEAR model. While this example is more complex, it grants a flexible interface to use
the LEAR model for real-time application.

.. literalinclude:: ../../examples/recalibrating_lear_flexible.py
  :language: python
