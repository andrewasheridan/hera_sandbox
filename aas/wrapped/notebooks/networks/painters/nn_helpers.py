import glob
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def log_dir(log_random_serial_number):
    master_log_path = '../../../logs/'
    log_num = len(glob.glob(master_log_path + log_random_serial_number + '*'))
    
    direc = master_log_path + log_random_serial_number
    if log_num > 0:
        # assumes less than 26 runs per graph
        direc += chr(ord('@') + log_num)
    return direc

def load_data(data_path, output_col = 'tau'):
    """Loads data from csv. Returns inputs and outputs as DataFrames"""
    df = pd.read_csv(data_path)
    
    inputs = df.drop(output_col, axis = 1)
    outputs = df[[output_col]]
    
    return [inputs, outputs]

def scale_data(data):
    """Returns scaled inputs & outputs, and their scalers."""
    input_scaler  = MinMaxScaler(feature_range = (0,1))
    output_scaler = MinMaxScaler(feature_range = (0,1))
    
    scaled_input  =  input_scaler.fit_transform(data[0])
    scaled_output = output_scaler.fit_transform(data[1])
    
    return [scaled_input, scaled_output, input_scaler, output_scaler]