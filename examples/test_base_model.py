import sys
# solution for relative imports in case realseries is not installed
sys.path.append('/home/cuipeng/realseries')
from realseries.models.base_rnn import rnn_base
from realseries.models.forecast import HNN
from realseries.utils.data import load_exp_data
import torch
model_test = rnn_base('LSTM', 6, [128, 64, 32], 2, 'tanh', 0.2, False)
model_test.cuda()
train_x = torch.randn(100, 5, 6).cuda()
y = model_test(train_x)
print(y.shape)
