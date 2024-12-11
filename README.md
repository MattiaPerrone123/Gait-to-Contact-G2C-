# Gait-to-Contact (G2C)

## Overview
This project aims to predict wear scars in total knee replacements (TKRs) settings using a transformer-CNN based encoder-decoder architecture. The methodology combines gait parameters as inputs and processes them to generate heatmaps representing wear scars, which are compared against ground truth using various metrics (MAPE, SSIM, NMI).

## Data
Input data are consists on multivariate time series including:
* anterior/posterior translation of the knee implant during gait 
* internal/external rotation of the knee implant during gait
* flexion/extension rotation of the knee implant during gait
* axial loading within the knee implant during gait
These data are not included in the repo and available on request.

## Repository Structure
* main.ipynb: Jupyter Notebook for running the pipeline.
* functions/: Contains modular Python functions organized as:
    * utilities.py: General utility functions
    * data_processing.py: Data preprocessing and handling
    * model.py: Model definition 
    * train_validate.py: Functions for training and validation
    * evaluate.py: Metrics and evaluation functions
    * metrics.py: Specific metrics calculation like SSIM, NMI, and MAPE
    * hp_tuning.py: Hyperparameter tuning utilities
 
      
* G2C - MedRxiv.pdf: MedRxiv preprint describing the methodology and results.
Usage


Data Preparation
This project processes input data and saves them to a folder before further analysis. Since the data is not included in the repository due to privacy concerns, users must provide their datasets. To integrate your data:
1. Place your raw input data in a folder (e.g., data/).
2. Update paths in main.ipynb:
    * path_data: Path to raw input data.
    * path_output: Path for saving processed data.


## Key Steps in the Pipeline
* Preprocessing: Processes raw gait parameters and saves them to an output folder.
* Model Training: Trains a continuous model for wear scar prediction, saving the best model in ONNX format.
* Evaluation: Computes evaluation metrics and visualizes predictions against ground truth.

  

## References
This work is a preprint and has not yet been peer-reviewed. Please cite the MedRxiv version as follows: DIRE CHE ORA è UNDER REVIEW

@article{your_reference,
  title={Wear Scar Prediction in Total Knee Replacements},
  author={Your Name et al.},
  journal={MedRxiv},
  year={2024}
}
