# -*- encoding: utf-8 -*-
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
# solution for relative imports in case realseries is not installed
sys.path.append(str(Path.cwd()))

from realseries.models.rnn import LSTMED
from realseries.utils.evaluation import point_metrics, adjust_metrics
from realseries.utils.data import load_splitted_RNN, load_NAB
from realseries.utils.visualize import plot_score, plot_anom
from realseries.utils.evaluation import point_metrics, adjust_metrics, thres_search
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = '6'
cwd = str(Path.cwd())

NAB = {
    'realKnownCause': [
        'rogue_agent_key_updown.csv',
        'nyc_taxi.csv',
    ]
}

dirname = 'realKnownCause'
filename = 'nyc_taxi.csv'
train_set, test_set = load_NAB(dirname, filename, fraction=0.5)

train_data, train_label = train_set.iloc[:, :-1], train_set.iloc[:, -1]
test_data, test_label = test_set.iloc[:, :-1], test_set.iloc[:, -1]

print('****************run lstm_enc_dec****************')
model_path = Path(cwd, 'snapshot/lstm_ed2', filename[:-4])
print('lstm_ed model path: {:}'.format(model_path))

lstm = LSTMED(epochs=60, seed=1111, batch_size=256, model_path=model_path)
lstm.fit(train_data, train_label, augment_length=50000)
score = lstm.detect(test_data)
precision, recall, f1, thres, pred_anom = thres_search(
    score,
    test_label,
    num_samples=500,
    beta=0.1,
    sampling='linear',
    adjust=False,
    delay=500)

# pred_anom = adjust_metrics(pred_anom,test_label,delay=1000)
plot_anom(
    test_set,
    pred_anom,
    score,
    if_save=True,
    name=Path(dirname, 'lstm_ed2' + filename.replace('csv', 'png')))