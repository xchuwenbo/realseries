from setuptools import setup, find_packages

setup(
    name='realseries', 
    version='0.0.1',
    description='a Python toolkit for dealing with time series data',
    url='https://realseries.readthedocs.io/',
    author='Realseries contributors',
    packages= find_packages(exclude=['examples', 'notebooks']),
    maintainer='Wenbo Hu',
    python_requires='>=3.6',
    install_requires=[
        "numpy>=1.13",
        "pandas>=0.25.3",
        "scikit-learn>=0.22",
        "scipy >=1.4.0",
        "pathlib >=1.0.1",
        "six >=1.13.0",
        "tqdm >=4.41.1",
        "statsmodels==0.10.2"
    ],
)
