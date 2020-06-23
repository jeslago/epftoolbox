.. _learex0:

==================
LEAR Examples
==================
This section contains examples on how to use the :ref:`LEAR` model. 

The :ref:`first example<learex1>` provides an easy-to-use interface for evaluating the LEAR model in a given test dataset. While this example lacks flexibility, it grants an simple interface to evalute LEAR models
in different datasets.

The :ref:`second example<learex2>` provides more flexible interface to perform recalibration and daily
forecasting with a LEAR model. While this example is more complex, it grants a flexible interface to use
the LEAR model for real-time application.

.. _learex1:

Easy recalibration
----------------------
.. literalinclude:: ../../examples/recalibrating_lear_simplified.py
  :language: python

.. _learex2:

Flexible recalibration
----------------------
.. literalinclude:: ../../examples/recalibrating_lear_flexible.py
  :language: python
