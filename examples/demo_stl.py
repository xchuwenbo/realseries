# -*- encoding: utf-8 -*-

# @Time    : 2020/02/14 13:08:20
# @Author  : ZHANG Xianrui

import os
import sys
import numpy as np
import pandas as pd

# solution for relative imports in case realseries is not installed
from Pathlib import Path
sys.path.append(str(Path.cwd()))
import matplotlib.pyplot as plt
import statsmodels.api as sm
from realseries.models.stl_decompose import STL
import warnings

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

dataset = sm.datasets.co2.load()
start = dataset.data['index'][0]
index = pd.date_range(start=start, periods=len(dataset.data), freq='W-SAT')
obs = pd.DataFrame(dataset.data['co2'], index=index, columns=['co2'])
obs.head()
obs = (obs.resample('D').mean().interpolate('linear'))
obs.head(100)
obs.plot()

# For example, with daily observations and large annual cycles, `period=365`.
#  For hourly observations with large daily cycles, `period=24`.
# Some inspection, and trial and error may be helpful.
aa=STL.calc_trend(obs.values.squeeze())
print(aa.shape)
model = STL()
decomp = model.fit(obs, period=365)
short_obs = obs.head(10000)
short_decomp = model.fit(short_obs, period=365)
fcast = model.forecast(stl=short_decomp, steps=8000)
print(fcast.head())

plt.figure()
plt.plot(obs, '--', label='truth')
plt.plot(short_obs, '--', label='obs')
plt.plot(short_decomp['trend'], ':', label='decomp.trend')
plt.plot(fcast, '-', label=fcast.columns[0])
plt.xlim('1970', '2004')
plt.ylim(330, 380)
plt.legend()
print(obs.shape,short_obs.shape,short_decomp['trend'].shape,fcast.shape)


fcast = model.forecast(short_decomp, steps=8000,seasonal=True)
print(fcast.head())
plt.figure()
plt.plot(obs, '--', label='truth')
plt.plot(short_obs, '--', label='obs')
plt.plot(short_decomp['trend'], ':', label='decomp.trend')
plt.plot(fcast, '-', label=fcast.columns[0])
plt.xlim('1970', '2004')
plt.ylim(330, 380)
plt.legend()
print(obs.shape,short_obs.shape,short_decomp['trend'].shape,fcast.shape)