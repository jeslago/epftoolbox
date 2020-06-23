
.. _dnnref:

==================
DNN
==================

The DNN model is a deep neural network tailored to electricity price forecasting whose input features and hyperparameters are optimized for each market without the need of expert knowledge. The DNN model was originally proposed by `Lago (2018) <https://doi.org/10.1016/j.apenergy.2018.02.069>`_ in a study where it was shown to obtain state-of-the-art results. Although more complex, it is often more accurate than individual :ref:`LEAR` models.

The module is built around the :py:class:`DNNModel <epftoolbox.models.DNNModel>` class. This class represents a basic DNN model based on keras and tensorflow. While the model can be used standalone to train and predict a DNN, it is intended to be used within the :py:class:`hyperparameter_optimizer <epftoolbox.models.hyperparameter_optimizer>` function and the :py:class:`DNN <epftoolbox.models.DNN>` class. 
These two elements represent the main two functionalities of the module.

In particular, the :py:class:`hyperparameter_optimizer <epftoolbox.models.hyperparameter_optimizer>` function provides an interface to optimize the optimal hyperparameter and features of the DNN model. Then, the :py:class:`DNN <epftoolbox.models.DNN>` class considers the output of the :py:class:`hyperparameter_optimizer <epftoolbox.models.hyperparameter_optimizer>` function, i.e. the set of optimal hyperparameters and features, and provides an interface to perform recalibration and new predictions. The class extends the functionality of the :class:`DNNModel` class by providing an interface to extract the best set of hyperparameters, and to perform recalibration before every prediction.


The module also includes the :py:func:`evaluate_dnn_in_test_dataset <epftoolbox.models.evaluate_dnn_in_test_dataset>` function. This function can be used as a simplified interface to evaluate a test period in a dataset that is built using a pandas DataFrame.

The library also includes several :ref:`dnnexo` to get users familiar with the syntax and capabilities of the model.


.. toctree::
   :maxdepth: 2
   
   DNNModel <dnn/DNNModel>
   hyperparameter_optimizer <dnn/hyperparameter_optimizer>
   DNN <dnn/DNN>
   evaluate_dnn_in_test_dataset <dnn/evaluate_dnn_in_test_dataset>
