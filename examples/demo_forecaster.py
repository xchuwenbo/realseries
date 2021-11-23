import sys
# solution for relative imports in case realseries is not installed
sys.path.append('/home/cuipeng/realseries')
import argparse
from realseries.models.base_rnn import rnn_base
from realseries.models.hnn import HNN
from realseries.models.mc_dropout import MC_dropout
from realseries.models.crmmd import CRMMD
from realseries.utils.data import load_exp_data

def hnn_demo(args, train_data, train_label, test_data, test_label, val_data, val_label, sc):
    print('The HNN model starts running')
    my_hnn = HNN(args.kernel_type, train_data.shape[-1],
             args.hiddens, args.prediction_window_size, args.activation, args.dropout_rate, epochs=args.epochs)
    my_hnn.fit(train_data, train_label, val_data, val_label,verbose=False)
    my_hnn.save_model()
    my_hnn.load_model()
    pi_low, pi_up, rmse, cali_error = my_hnn.evaluation_model(
        sc, test_data, test_label, t=1, confidence = args.confidence)

    result = {
        'pis_low': pi_low,
        'pis_up': pi_up,
        'rmse': rmse,
        'cali_error': cali_error
    }
    return result

def mc_demo(args, train_data, train_label, test_data, test_label, val_data, val_label, sc):
    print('The MC_dropout model starts running')
    my_mc_model = MC_dropout(args.kernel_type, train_data.shape[-1],
             args.hiddens, args.prediction_window_size, args.activation, args.dropout_rate, epochs=args.epochs, variance=False)
    my_mc_model.fit(train_data, train_label, val_data, val_label,verbose=False)
    my_mc_model.save_model()
    my_mc_model.load_model()
    pi_low, pi_up, rmse, cali_error = my_mc_model.evaluation_model(
        sc, test_data, test_label, t=1, confidence = args.confidence,mc_times=1000)

    result = {
        'pis_low': pi_low,
        'pis_up': pi_up,
        'rmse': rmse,
        'cali_error': cali_error
    }
    return result

def crmmd_demo(args, train_data, train_label, test_data, test_label, val_data, val_label, sc):
    print('The CRMMD model starts running')
    my_crmmd = CRMMD(args.kernel_type, train_data.shape[-1],
             args.hiddens, args.prediction_window_size, args.activation, args.dropout_rate, epochs_hnn=args.epochs, epochs_mmd=args.epochs_mmd)
    my_crmmd.fit(train_data, train_label, val_data, val_label,verbose=False)
    my_crmmd.save_model()
    my_crmmd.load_model()
    pi_low, pi_up, rmse, cali_error = my_crmmd.evaluation_model(
        sc, test_data, test_label, t=1, confidence = args.confidence)

    result = {
        'pis_low': pi_low,
        'pis_up': pi_up,
        'rmse': rmse,
        'cali_error': cali_error
    }
    return result
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run forecaster.')
    parser.add_argument('--kernel_type', default='LSTM')
    parser.add_argument('--activation', default='tanh')
    parser.add_argument('--dropout_rate', type=float, default=0.3)
    parser.add_argument('--window_size', type=int, default=10)
    parser.add_argument('--prediction_window_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--epochs_mmd', type=int, default=200)
    parser.add_argument('--fractions', default=[0.6,0.2,0.2])
    parser.add_argument('--hiddens', default=[128,32])
    parser.add_argument('--data_name', default='air_quality')
    parser.add_argument('--confidence', type=int, default=90)
    args = parser.parse_args(args=[])
    train_data, train_label, test_data, test_label, val_data, val_label, sc = load_exp_data(
        args.data_name, args.window_size, args.prediction_window_size, args.fractions)
    
    ''' result = hnn_demo(args, train_data, train_label, test_data,
                  test_label, val_data, val_label, sc) '''
    result = mc_demo(args, train_data, train_label, test_data,
                  test_label, val_data, val_label, sc)
    
    """ result = crmmd_demo(args, train_data, train_label, test_data,
                  test_label, val_data, val_label, sc) """
    
    print('rmse:{}, calibration error:{}'.format(result['rmse'],result['cali_error']))

