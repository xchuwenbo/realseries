API Cheetsheet
==============

Model
-----

- **IsolationForest**

  * :func:`realseries.models.iforest.IForest.fit`: Fit Isolation Forest. y is
    ignored.
  * :func:`realseries.models.iforest.IForest.detect`: Predict the score of a
    sample being anomaly by the detector. The anomaly score is returned.

- **LSTM_dynamic**

  * :func:`realseries.models.lstm_dynamic.LSTM_dynamic.fit`: Fit LSTM model. y
    is ignored.
  * :func:`realseries.models.lstm_dynamic.LSTM_dynamic.detect`: Predict the
    score of a sample being anomaly by the dynamic method. The anomaly sequence
    and score is returned.

- **Luminol**

  * :func:`realseries.models.lumino.Lumino.detect`: Predict the score of a
    sample being anomaly by the detector. The anomaly score is returned.

- **Random cut forest**

  * :func:`realseries.models.rcforest.RCForest.detect`: Predict the score of a
    sample being anomaly by the detector. The anomaly score is returned.

- **LSTM encoder decoder**

  * :func:`realseries.models.rnn.LSTMED.fit`: Fit LSTM. y is ignored.
  * :func:`realseries.models.rnn.LSTMED.detect`: Predict the score of a
    sample being anomaly by the LSTM. The anomaly score is returned.

- **SeqVL**

  * :func:`realseries.models.seqvl.SeqVL.fit`: Fit detector. y is ignored
    in unsupervised methods.
  * :func:`realseries.models.seqvl.SeqVL.detect`: Predict the score of a
    sample being anomaly by the detector. The anomaly score is returned.

- **SR_CNN**

  * :func:`realseries.models.srcnn.SR_CNN.fit`: Fit CNN model.
  * :func:`realseries.models.srcnn.SR_CNN.detect`: Predict the score of a
    sample being anomaly by the CNN. The anomaly score is returned.

- **VAE_AD**

  * :func:`realseries.models.vae_ad.VAE_AD.fit`: Fit detector. y is ignored
    in unsupervised methods.
  * :func:`realseries.models.vae_ad.VAE_AD.detect`: Predict the score of a
    sample being anomaly by the detector. The anomaly score is returned.

- **STL**

  * :func:`realseries.models.stl.STL.fit`: Fit STL model. y is ignored
    in unsupervised methods.
  * :func:`realseries.models.stl.STL.forecast`: Forecast the later value of a
    sequence. The array is returned.

- **Granger Causality**

  * :func:`realseries.models.GC.GC.detect`: Granger Causality detector, which is channel-level.
  * :func:`realseries.models.DWGC.DWGC.detect`: Dynamic Window-level Granger Causality detector.

See base class definition in :class:`realseries.models.base`.

Data
----

The following functions are used for raw data loading easily.

* :func:`realseries.utils.data.load_NAB`: Load data in the `NAB_data` diary.
  Train DataFrame and Test DataFrame with labels are returned.

* :func:`realseries.utils.data.load_Yahoo`: Load data in the `Yahoo_data`
  diary. Train DataFrame and Test DataFrame with labels are returned.

* :func:`realseries.utils.data.load_split_NASA`: Load data in the `NASA`
  diary. Train DataFrame and Test DataFrame with labels are returned.

Visualize
---------

The following functions are used plotting raw data and predicted result.

* :func:`realseries.utils.visualize.plot_anom`: The parameters mainly include
  ``pd_data_label``, ``pred_anom`` and ``pred_score``. ``pd_data_label`` is the
  :func:`pandas.DataFrame` with data and label, ``pred_anom`` is the array with
  predicted label, and ``pred_score`` is the corresponding anomaly score.

* :func:`realseries.utils.visualize.plot_mne`: The parameters mainly include
  ``X, scalings, ch_types, color``. If ``X`` is the array and last column as label.
  We set label column to different ``ch_type``, so it will show different color in
  the figure.

