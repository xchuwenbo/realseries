.. RealSeries documentation master file, created by
   sphinx-quickstart on Sun Mar  1 23:13:58 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to RealSeries's documentation!
======================================

RealSeries is a comprehensive **out-of-the-box Python toolkit** for various tasks, including :ref:`Anomaly Detection`, :ref:`Granger causality` and :ref:`Forecast with Uncertainty`, of dealing with :ref:`Time Series Datasets`. 



RealSeries has the following features:

* Unified APIs, detailed documentation, easy-to-follow examples and straightforward visualizations.
* All-levels of models, including simple thresholds, classification-based models, and deep (Bayesian) models.

.. warning::

   RealSeries supports **Python 3 ONLY**. See `here <https://wiki.python.org/moin/Python2orPython3>`_ to check why.

----

**API Demo**:

RealSeries uses the `sklearn-style API <https://scikit-learn.org/stable/modules/classes.html>`_ and is as easy as

.. code-block:: python
    :linenos:

    # train the SR-CNN detector
    from realseries.models.sr_2 import SR_CNN
    sr_cnn = SR_CNN(model_path)
    sr_cnn.fit(train_data)
    score = sr_cnn.detect(test_data, test_label)
    
----

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   pages/installation
   pages/examples

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   pages/cheetsheet
   pages/realseries
   pages/datasets

.. toctree::
   :maxdepth: 2
   :caption: RealSeries 101

   pages/anomaly_detection
   pages/Granger_causality
   pages/forecast_with_uncertainty

.. toctree::
   :maxdepth: 2
   :caption: Contribution

   pages/contribution
   pages/how-to-contribute


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
