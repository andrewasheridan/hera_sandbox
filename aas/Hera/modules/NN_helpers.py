import glob
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import os, sys, io
sys.path.insert(1, os.path.join(sys.path[0], '/Users/andrew/Documents/Lab/aas/Hera/modules'))
import NN_helpers

def log_dir(log_id, lod_directory = 'logs/'):
    
    log_id = str(log_id)
    # how many logs already have this log_ID
    log_num = len(glob.glob(lod_directory + log_id + '*'))
    
    path = lod_directory + log_id
    
    # append A->Z if log already exists
    if log_num > 0:
        path += chr(ord('@') + log_num)
        
    print('Log Directory: %s' %path)  
    return path

def load_data(path):
    """Loads data from npz. Returns inputs and outputs"""
    a = np.load(path)
    
    inputs = a['arr_1']
    outputs = a['arr_0']
    
    return [inputs, outputs]

def scale_data(data):
    """Returns scaled inputs & outputs, and their scalers."""
    
    input_scaler  = MinMaxScaler(feature_range = (0,1))
    output_scaler = MinMaxScaler(feature_range = (0,1))
    
    scaled_input  =  input_scaler.fit_transform(data[0])
    scaled_output = output_scaler.fit_transform(data[1])
    
    return [scaled_input, scaled_output, input_scaler, output_scaler]

def pre_process_data(data_path, testing_data_percentage):
    data = NN_helpers.load_data(data_path)
    scaled_input, scaled_output, input_scaler, output_scaler = NN_helpers.scale_data(data)
    X_train, X_test, y_train, y_test = train_test_split(scaled_input, scaled_output,
                                                    test_size = testing_data_percentage,
                                                    random_state = np.random.seed(int(time.time())))
    
    return (X_train, X_test, y_train, y_test, input_scaler, output_scaler)



def gen_plot(predicted_values, actual_values, output_scaler):
    """Create a prediction plot and save to byte string."""
    
    prediction_unscaled = output_scaler.inverse_transform(predicted_values)
    actual_unscaled = output_scaler.inverse_transform(actual_values)

    sorting_idx = np.argsort(actual_unscaled.T[0])
        
    fig, ax = plt.subplots(figsize = (5, 3), dpi = 144)

    ax.plot(prediction_unscaled.T[0][sorting_idx],
            linestyle = 'none', marker = '.', markersize = 1,
            color = 'darkblue')
    
    ax.plot(actual_unscaled.T[0][sorting_idx],
            linestyle = 'none', marker = '.', markersize = 1, alpha = 0.50,
            color = '#E50000')       
    
    ax.set_title('std: %.2f' %np.std(prediction_unscaled.T[0][sorting_idx] - actual_unscaled.T[0][sorting_idx]))
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi = 144)
    plt.close(fig)
    buf.seek(0)

    return buf.getvalue()

