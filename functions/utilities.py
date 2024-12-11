
import os
import re
import numpy as np
import pandas as pd

from .interpolate import interpolate, plot_results, read_data, save_interpolated_data



def create_2d_dict(data):
    """
    Create a 2D dictionary from the given data using the position values directly as indices
    """
    x_indices=pd.factorize(data['x'])[0]
    y_indices=pd.factorize(data['y'])[0]

    matrix_dict={}
    for i in range(len(data)):
        matrix_dict[(x_indices[i], y_indices[i])]=data['value'].iloc[i]

    return matrix_dict


def dict_to_compact_array(matrix_dict):
    """
    Convert a 3D matrix dictionary to a compact 2D numpy array
    """

    indices_and_values=np.array([(x, y, value) for (x, y), value in matrix_dict.items()])
    return indices_and_values


def sort_columns(column):
    match=re.match(r"([a-z]+)([0-9]+)", column)
    if match:
        return match.group(1), int(match.group(2))
    else:
        return column, 0



def extract_column_info(filename):
    """
    Extract the prefix and number from a filename and return a standardized column name
    """
    match=re.match(r"([a-z]+)([0-9]+)", filename, re.I)
    if match:
        prefix, number=match.groups()
        prefix=prefix.lower()
        column_name=f"{prefix}{int(number)}"
        return column_name, prefix
    return None, None


def load_last_column(filepath):
    """
    Load the last column of a CSV file as a Series
    """
    return pd.read_csv(filepath, header=None).iloc[:, -1]



def process_output_data(columns_output, path_data, save_suffix, m=100, n=100, plot=True, save=True):
    """Process a list of output column names by reading, interpolating, and optionally plotting and saving the results"""
    for col_name in columns_output:
        data_filepath=f"{path_data}/{col_name}.txt"
        data_np=read_data(data_filepath)

        xi, yi, zi=interpolate(data_np, m, n)

        if plot:
            plot_results(xi, yi, zi, data_np)

        if save:
            save_interpolated_data(xi, yi, zi, col_name, n, m, save_suffix)

def load_and_stack_processed_images(path_output, m, n, file_suffix):
    """Load processed image files from a directory, normalize them, and stack into a 4D Numpy array"""
    file_list_output=os.listdir(path_output)
    file_list_output=[f for f in file_list_output if file_suffix in f]
    file_list_output.sort()

    all_data_output_list=[]

    for file_interp in file_list_output:
        data_processed=pd.read_csv(os.path.join(path_output, file_interp), header=None, names=['x', 'y', 'value'])

        matrix_dict_2d=create_2d_dict(data_processed)

        compact_array_2d=dict_to_compact_array(matrix_dict_2d)

        image=np.zeros((n, m, 1))

        for x_val, y_val, value in compact_array_2d:
            image[int(x_val), int(y_val), 0]=value

        max_value=np.max(image)
        if max_value!=0:
            image=image / max_value
        image=np.rot90(image)

        all_data_output_list.append(image)

    all_data_output=np.stack(all_data_output_list)
    all_data_output=all_data_output.transpose(0, 3, 1, 2)

    return all_data_output




    
