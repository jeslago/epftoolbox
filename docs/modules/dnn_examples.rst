.. _dnnexo:

==================
DNN Examples
==================
This section contains examples on how to use the :ref:`DNN` model. 

The :ref:`first example<dnnex1>` introduces the hyperparameter optimization and feature selection
procedure of the :ref:`DNN` model. 

The :ref:`second example<dnnex2>` provides an easy-to-use interface for evaluating the LEAR model in a given test dataset. While this example lacks flexibility, it grants an simple interface to evalute LEAR models
in different datasets. It is important to note that this example assumes that a hyperparameter optimization
procedure has already been performed.

The :ref:`third example<dnnex3>` provides more flexible interface to perform recalibration and daily
forecasting with a LEAR model. While this example is more complex, it grants a flexible interface to use
the LEAR model for real-time application. It is important to note that this example assumes that a hyperparameter optimization procedure has already been performed.

.. _dnnex1:

Hyperparameter optimization
----------------------
.. literalinclude:: ../../examples/optimizing_hyperparameters_dnn.py
  :language: python

.. _dnnex2:

Easy recalibration
----------------------
.. literalinclude:: ../../examples/recalibrating_dnn_simplified.py
  :language: python

.. _dnnex3:

Flexible recalibration
----------------------
.. literalinclude:: ../../examples/recalibrating_dnn_flexible.py
  :language: python
