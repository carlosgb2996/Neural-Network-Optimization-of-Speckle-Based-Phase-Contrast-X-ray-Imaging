# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import GPUtil

def load_displacement(file_path):
    """
    Loads a displacement map from a CSV file.
    
    Parameters:
    - file_path: Path to the CSV file.
    
    Returns:
    - displacement: 2D numpy array or None if loading fails.
    """
    try:
        displacement = np.loadtxt(file_path, delimiter=',')
        return displacement
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def calculate_rmse(pred_disp, gt_disp):
    """
    Calculates RMSE between predicted and ground truth displacements.
    
    Parameters:
    - pred_disp: 2D numpy array of predicted displacements.
    - gt_disp: 2D numpy array of ground truth displacements.
    
    Returns:
    - rmse: RMSE value.
    """
    error = pred_disp - gt_disp
    mse = np.mean(error ** 2)
    rmse = np.sqrt(mse)
    return rmse

def calculate_bias(pred_disp, gt_disp):
    """
    Calculates Bias between predicted and ground truth displacements.
    
    Parameters:
    - pred_disp: 2D numpy array of predicted displacements.
    - gt_disp: 2D numpy array of ground truth displacements.
    
    Returns:
    - bias: Bias value.
    """
    error = pred_disp - gt_disp
    bias = np.mean(error)
    return bias

def calculate_r2(pred_disp, gt_disp):
    """
    Calculates R^2 (Coefficient of Determination) between predicted and ground truth displacements.
    
    Parameters:
    - pred_disp: 2D numpy array of predicted displacements.
    - gt_disp: 2D numpy array of ground truth displacements.
    
    Returns:
    - r2: R^2 value.
    """
    ss_res = np.sum((pred_disp - gt_disp) ** 2)
    ss_tot = np.sum((gt_disp - np.mean(gt_disp)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

def calculate_r2_general(pred_disp_x, pred_disp_y, gt_disp_x, gt_disp_y):
    """
    Calculates a general R^2 value considering both x and y displacements.
    
    Parameters:
    - pred_disp_x: 2D numpy array of predicted displacements in x-direction.
    - pred_disp_y: 2D numpy array of predicted displacements in y-direction.
    - gt_disp_x: 2D numpy array of ground truth displacements in x-direction.
    - gt_disp_y: 2D numpy array of ground truth displacements in y-direction.
    
    Returns:
    - r2_general: General R^2 value.
    """
    # Flatten the arrays to treat all displacements together
    pred_all = np.concatenate((pred_disp_x.flatten(), pred_disp_y.flatten()))
    gt_all = np.concatenate((gt_disp_x.flatten(), gt_disp_y.flatten()))
    
    ss_res = np.sum((pred_all - gt_all) ** 2)
    ss_tot = np.sum((gt_all - np.mean(gt_all)) ** 2)
    r2_general = 1 - (ss_res / ss_tot)
    return r2_general

def get_gpu_info():
    """
    Retrieves information about the GPU being used.
    
    Returns:
    - gpu_info: String containing GPU details or a message if no GPU is found.
    """
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]  # Selecciona la primera GPU disponible
        gpu_info = (f"GPU ID: {gpu.id}, Name: {gpu.name}, "
                    f"Temperature: {gpu.temperature} C, "
                    f"Memory Usage: {gpu.memoryUsed}/{gpu.memoryTotal} MB, "
                    f"Load: {gpu.load*100}%")
    else:
        gpu_info = "No GPU found."
    return gpu_info

def main(pred_dir, gt_dir, output_csv):
    """
    Calculates RMSE, Bias, and R^2 metrics for all displacement pairs.
    
    Parameters:
    - pred_dir: Directory containing predicted displacement files.
    - gt_dir: Directory containing ground truth displacement files.
    - output_csv: Path to the output CSV file for results.
    """
    # Mostrar información de la GPU
    gpu_info = get_gpu_info()
    print(f"=== GPU Information ===\n{gpu_info}\n")
    
    # Find all predicted displacement x files
    pred_x_pattern = os.path.join(pred_dir, 'Def*_disp_x.csv')
    pred_x_files = sorted(glob.glob(pred_x_pattern))
    
    if not pred_x_files:
        print(f"No files found matching the pattern {pred_x_pattern}")
        return
    
    results = []
    
    # Utilizar tqdm para mostrar una barra de progreso
    for pred_x_file in tqdm(pred_x_files, desc="Processing displacement files"):
        # Extract the identifier XXX_YY from the filename
        filename = os.path.basename(pred_x_file)
        # Assuming the format is DefXXX_YY_disp_x.csv
        try:
            parts = filename.split('_')
            XXX = parts[0][3:]  # Remove 'Def' to get XXX
            YY = parts[1]
            identifier = f"{XXX}_{YY}"
        except IndexError:
            print(f"Unexpected filename format: {filename}")
            continue
        
        # Construct paths to the corresponding y predicted and ground truth files
        pred_y_file = os.path.join(pred_dir, f'Def{identifier}_disp_y.csv')
        gt_x_file = os.path.join(gt_dir, f'Dispx{identifier}.csv')
        gt_y_file = os.path.join(gt_dir, f'Dispy{identifier}.csv')
        
        # Check if all files exist
        if not all([os.path.exists(pred_y_file), os.path.exists(gt_x_file), os.path.exists(gt_y_file)]):
            print(f"Missing files for identifier {identifier}. Skipping this pair.")
            continue
        
        # Load displacements
        disp_pred_x = load_displacement(pred_x_file)
        disp_pred_y = load_displacement(pred_y_file)
        disp_gt_x = load_displacement(gt_x_file)
        disp_gt_y = load_displacement(gt_y_file)
        
        # Verify successful loading
        if any(x is None for x in [disp_pred_x, disp_pred_y, disp_gt_x, disp_gt_y]):
            print(f"Error loading some files for {identifier}. Skipping this pair.")
            continue
        
        # Calculate metrics
        rmse_x = calculate_rmse(disp_pred_x, disp_gt_x)
        rmse_y = calculate_rmse(disp_pred_y, disp_gt_y)
        bias_x = calculate_bias(disp_pred_x, disp_gt_x)
        bias_y = calculate_bias(disp_pred_y, disp_gt_y)
        r2_x = calculate_r2(disp_pred_x, disp_gt_x)
        r2_y = calculate_r2(disp_pred_y, disp_gt_y)
        r2_general = calculate_r2_general(disp_pred_x, disp_pred_y, disp_gt_x, disp_gt_y)
        
        # Store results
        results.append({
            'Identifier': identifier,
            'RMSE_x': rmse_x,
            'RMSE_y': rmse_y,
            'Bias_x': bias_x,
            'Bias_y': bias_y,
            'R2_x': r2_x,
            'R2_y': r2_y,
            'R2_general': r2_general
        })
    
    if not results:
        print("No metrics were calculated for any displacement pairs.")
        return
    
    # Create a pandas DataFrame
    df_results = pd.DataFrame(results)
    
    # Save results to a CSV file
    df_results.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}\n")
    
    # Calculate summary statistics
    mean_rmse_x = df_results['RMSE_x'].mean()
    std_rmse_x = df_results['RMSE_x'].std()
    mean_rmse_y = df_results['RMSE_y'].mean()
    std_rmse_y = df_results['RMSE_y'].std()
    
    mean_bias_x = df_results['Bias_x'].mean()
    std_bias_x = df_results['Bias_x'].std()
    mean_bias_y = df_results['Bias_y'].mean()
    std_bias_y = df_results['Bias_y'].std()
    
    mean_r2_x = df_results['R2_x'].mean()
    std_r2_x = df_results['R2_x'].std()
    mean_r2_y = df_results['R2_y'].mean()
    std_r2_y = df_results['R2_y'].std()
    mean_r2_general = df_results['R2_general'].mean()
    std_r2_general = df_results['R2_general'].std()
    
    # Display summary statistics
    print("=== RMSE Statistics ===")
    print(f"RMSE_x: Mean = {mean_rmse_x:.6f}, Standard Deviation = {std_rmse_x:.6f}")
    print(f"RMSE_y: Mean = {mean_rmse_y:.6f}, Standard Deviation = {std_rmse_y:.6f}\n")
    
    print("=== Bias Statistics ===")
    print(f"Bias_x: Mean = {mean_bias_x:.6f}, Standard Deviation = {std_bias_x:.6f}")
    print(f"Bias_y: Mean = {mean_bias_y:.6f}, Standard Deviation = {std_bias_y:.6f}\n")
    
    print("=== R^2 Statistics ===")
    print(f"R2_x: Mean = {mean_r2_x:.6f}, Standard Deviation = {std_r2_x:.6f}")
    print(f"R2_y: Mean = {mean_r2_y:.6f}, Standard Deviation = {std_r2_y:.6f}")
    print(f"R2_general: Mean = {mean_r2_general:.6f}, Standard Deviation = {std_r2_general:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate RMSE, Bias, and R^2 metrics for predicted displacements.')
    parser.add_argument('--pred_dir', type=str, default='/hpcfs/home/fisica/s.naranjob/StrainNet/StrainNet/output_inference/',
                        help='Directory containing predicted displacement files.')
    parser.add_argument('--gt_dir', type=str, default='/hpcfs/home/fisica/s.naranjob/StrainNet/Dataset/Test_Data/',
                        help='Directory containing ground truth displacement files.')
    parser.add_argument('--output_csv', type=str, default='metrics_results.csv',
                        help='CSV file to save the results.')
    
    args = parser.parse_args()
    
    main(args.pred_dir, args.gt_dir, args.output_csv)