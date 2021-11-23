Installation
============

RealSeries is still under development. Before the first stable release, you can install the RealSeries from source.

If you use RealSeries temporarily, you may clone the code from `GitHub repository <https://github.com/RealSeries/realseries>`_ and add it to your Python path:

    .. code-block:: python

        git clone https://github.com/RealSeries/realseries.git
        cd RealSeries # change current work directory to ./RealSeries
        python
        >>> import sys,os
        >>> sys.path.append(os.getcwd()) # add ./RealSeries to sys.path
        >>> import realseries
        >>> from realseries.models.iforest import IForest

Alternatively, you can install it:

    .. code-block:: python

        git clone https://github.com/RealSeries/realseries.git # clone
        cd RealSeries
        pip install .
        python
        >>> import realseries
        >>> from realseries.models.iforest import IForest