# realseries

[![Documentation Status](https://readthedocs.org/projects/realseries/badge/?version=latest)](https://realseries.readthedocs.io/en/latest/?badge=latest)

**RealSeries** is a comprehensive out-of-the-box Python toolkit for various tasks, including Anomaly Detection, Granger causality and Forecast with Uncertainty, of dealing with Time Series Datasets.

**RealSeries** has the following features:

* Unified APIs, detailed documentation, easy-to-follow examples and straightforward visualizations.
* All-levels of models, including simple thresholds, classification-based models, and deep (Bayesian) models.

**RealSeries** also considers the **icing-on-the-cake** functions, including:
* Bayesian Time series forecast/detect to get **uncertainty**.
* Model **interpretations** for the prediction/detection including **causality**.

**API Demo**:

RealSeries uses the [sklearn-style API](https://scikit-learn.org/stable/modules/classes.html) and is as easy as

```python
# train the SR-CNN detector
from realseries.models.sr_2 import SR_CNN
sr_cnn = SR_CNN(model_path)
sr_cnn.fit(train_data)
score = sr_cnn.detect(test_data, test_label)
```
## Installation
RealSeries is still under development. Before the first stable release (1.0), you can install the RealSeries from source.

If you use RealSeries temporarily, you may clone the code and add it to your Python path:
```python
git clone https://github.com/RealSeries/realseries.git
cd RealSeries # change current work directory to ./RealSeries
python
>>> import sys,os
>>> sys.path.append(os.getcwd()) # add ./RealSeries to sys.path
>>> import realseries
>>> from realseries.models.iforest import IForest
```

Alternatively, you can install it:
```python
git clone https://github.com/RealSeries/realseries.git # clone
cd RealSeries
pip install .
python
>>> import realseries
>>> from realseries.models.iforest import IForest
```

>[!WARNING]
> RealSeries supports Python3 (=3.7.11) **ONLY**.

### Dependencies
Realseries has several deep learning models like sr_cnn, lstm_encoder_decoder,
   which are implemented by pytorch. However, we do **NOT** install dependencies like **pytorch**
   automatically for you. If you want to use neural-net based models, see
   [Installing Pytorch](https://pytorch.org/) for installation
   Similarly, models depending on **rrcf** and **luminol** would **NOT**
   installed by default.



#### Required Dependencies:

* Python=3.7.11
* numpy>=1.13
* pandas>=0.25.3
* scikit-learn>=0.22
* scipy >=1.4.0
* pathlib >=1.0.1
* six >=1.13.0
* tqdm >=4.41.1
* setuptools==49.1
* statsmodels==0.10.2

#### Optional Dependencies:

>[!WARNING]
> We do **NOT** install all dependencies for you. Instead you should install dependencies by yourself for any models you want.

* pytorch (required for LSTM_AutoEncoder)
* rrcf>=0.3.2 (required for Random cut forest)
* luminol>=0.4 (required for luminol)
* more_itertools (required for LSTM_dynamic)
* mne (required for visualization examples)
* matplotlib (required for plotting examples)

## Documentation
* [Tutorials and API Docs](https://realseries.readthedocs.io/) 
* [Tutorials and API Docs Source Code](docs/) (Need to build with [Sphinx](https://www.sphinx-doc.org/en/master/))
* [Gitlab Jupyter Notebooks](notebooks/)
* [Gitlab Python Examples](examples/)

## Contribution and Contact
* The core development members include
  * [Wenbo Hu](https://ml.cs.tsinghua.edu.cn/~wenbo/)
  * Xianrui Zhang
  * Wenkai Li
  * Zhiheng Zhang

If you have any questions, please leave issues.

## Acknowkledgement
This project directly used many open-source libs:
* pytorch,
* luminol,
* sklearn,
* statsmodel.

Please leave an issue or send email to [Wenbo Hu](https://ml.cs.tsinghua.edu.cn/~wenbo/) if your project wants to show in the list or does not want to be used.
