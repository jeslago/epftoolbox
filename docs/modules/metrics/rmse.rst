==================
RMSE
==================
In the field of electricity price forecasting, one of the most widely used metrics to measure the accuracy of point forecasts is the root mean square erro (RMSE):

.. math::

    \begin{align}
        \mathrm{RMSE} &= \sqrt{\frac{1}{N}\sum_{k=1}^{N}(p_k-\hat{p}_k)^2},\\
    \end{align}

This metric computes the square root of the average of the  square errors between the predicted prices and the real prices. Predictive models that minimize the RMSE lead to predictions of the mean of the prices. 

Despite its popularity,RMSE is not always very informative as absolute errors are hard to compare between different datasets. In addition, it has the disadvantage of not representing accurately the underlying problem of electricity price forecasting as electricity costs often depend linearly on prices but the RMSE is based on squared errors. 


epftoolbox.evaluation.RMSE
---------------------------

.. autofunction:: epftoolbox.evaluation.RMSE