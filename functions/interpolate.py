import os
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from shapely.plotting import plot_polygon
from alphashape import alphashape


def read_data(file_path):
    """Read data from a file and return as a NumPy array"""
    with open(file_path, 'r') as file:
        lines=file.readlines()
    data=[list(map(float, line.strip().split(','))) for line in lines]
    return np.array(data)


def interpolate(data, n, m):
    """Interpolate data onto a uniform grid"""
    x, y, z, values=data[:, 0], data[:, 1], data[:, 2], data[:, 3]

    xi=np.linspace(min(x), max(x), n)
    yi=np.linspace(min(y), max(y), m)
    xi, yi=np.meshgrid(xi, yi)

    zi=griddata((x, y), values, (xi, yi), method='linear')
    zi=np.nan_to_num(zi)

    return xi, yi, zi


def plot_results(xi, yi, zi, original_data):
    """Plot interpolated results and the alpha shape of original data"""
    fig, ax=plt.subplots()
    contour=ax.contourf(xi, yi, zi, levels=20, cmap='viridis')
    plt.colorbar(contour, ax=ax, label='Interpolated Values')

    alpha_shape=alphashape(original_data[:, :2], alpha=0.85)
    plot_polygon(alpha_shape)

    plt.title('Interpolation of Data Points onto Uniform Grid')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()


def get_file_path():
    """Open a file dialog to select a file and return its path"""
    root=Tk()
    root.withdraw()
    file_path=filedialog.askopenfilename(title="Select Data File", filetypes=[("Text files", "*.txt")])
    root.destroy()
    return file_path


def save_interpolated_data(xi, yi, zi, original_file_name, n, m, output_folder):
    """Save interpolated data to a file"""
    output_file_name=f"{os.path.splitext(original_file_name)[0]}_{n}x{m}_interpolated.txt"
    output_path=os.path.join(output_folder, output_file_name)
    output_data=np.column_stack((xi.flatten(), yi.flatten(), zi.flatten()))
    np.savetxt(output_path, output_data, delimiter=',', fmt='%.18e', comments='')
