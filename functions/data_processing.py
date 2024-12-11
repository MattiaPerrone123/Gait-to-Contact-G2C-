
import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from .utilities import sort_columns, extract_column_info, load_last_column



def process_inputs_and_outputs(path_data, file_list, inputs, output):
    """Process the given file list, separating input and output data into respective DataFrames"""
    all_inputs={key:pd.DataFrame() for key in inputs}
    columns_inputs={key:[] for key in inputs}

    all_output=pd.DataFrame()
    columns_output=[]

    for filename in file_list:
        column_name, prefix=extract_column_info(filename)
        if not column_name:
            continue

        filepath=os.path.join(path_data, filename)
        data_col=load_last_column(filepath)

        if any(key in filename for key in inputs):
            for key in inputs:
                if key in filename:
                    all_inputs[key]=pd.concat([all_inputs[key], data_col], axis=1, ignore_index=True)
                    columns_inputs[key].append(column_name)
                    break
        elif output in filename:
            all_output=pd.concat([all_output, data_col], axis=1, ignore_index=True)
            columns_output.append(column_name)

    for key in inputs:
        all_inputs[key].columns=columns_inputs[key]
    all_output.columns=columns_output

    return all_inputs, all_output, columns_output

def process_files(path_data, inputs, output):
    """Process input and output files, sorting and organizing them into DataFrames"""
    file_list=os.listdir(path_data)
    file_list.sort(key=sort_columns)

    return process_inputs_and_outputs(path_data, file_list, inputs, output)

def get_files_with_keyword(path_data, keyword):
    """Return a sorted list of files in 'path_data' that contain 'keyword'"""
    file_list=os.listdir(path_data)
    file_list.sort(key=sort_columns)
    return [f for f in file_list if keyword in f]


def prepare_input_data(all_input_ap, all_input_fe, all_input_force, all_input_ie, inputs):
    """Prepare the 3D NumPy array of input data with shape (n_obs, time_steps, n_features)"""
    n_obs=all_input_ap.shape[1]
    time_steps=all_input_ap.shape[0]
    n_features=len(inputs)

    all_data_input=np.zeros((n_obs, time_steps, n_features))
    all_data_input[:, :, 0]=np.transpose(all_input_ap)
    all_data_input[:, :, 1]=np.transpose(all_input_fe)
    all_data_input[:, :, 2]=np.transpose(all_input_force)
    all_data_input[:, :, 3]=np.transpose(all_input_ie)

    return all_data_input


def scale_features(X_train, X_test):
    """Scale features for train and test sets using MinMaxScaler"""
    X_train_scaled=np.empty_like(X_train)
    X_test_scaled=np.empty_like(X_test)
    scaler=MinMaxScaler()

    for i in range(X_train.shape[-1]):
        X_train_scaled[:, :, i]=scaler.fit_transform(X_train[:, :, i])
        X_test_scaled[:, :, i]=scaler.transform(X_test[:, :, i])

    return X_train_scaled, X_test_scaled





def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=4):
    """Create DataLoader objects for training and testing datasets"""
    train_dataset=TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
    test_dataset=TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).float())

    train_dataloader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader=DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader

def drop_first_time_step(X):
    """Drop the first time step from the input data"""
    return X[:, 1:, :]

def prepare_train_test_dataloaders(all_input_ap, all_input_fe, all_input_force, all_input_ie, all_data_output, inputs, test_size=0.2, random_state=15, batch_size=4):
    """Prepare DataLoaders for training and testing by processing input and output data"""
    X=prepare_input_data(all_input_ap, all_input_fe, all_input_force, all_input_ie, inputs)

    y=all_data_output

    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=test_size, random_state=random_state)

    X_train=drop_first_time_step(X_train)
    X_test=drop_first_time_step(X_test)

    X_train_scaled, X_test_scaled=scale_features(X_train, X_test)

    train_dataloader, test_dataloader=create_dataloaders(X_train_scaled, y_train, X_test_scaled, y_test, batch_size)

    return train_dataloader, test_dataloader













    
