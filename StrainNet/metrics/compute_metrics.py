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

def calculate_mae(pred_disp, gt_disp):
    """
    Calculates MAE between predicted and ground truth displacements.
    
    Parameters:
    - pred_disp: 2D numpy array of predicted displacements.
    - gt_disp: 2D numpy array of ground truth displacements.
    
    Returns:
    - mae: MAE value.
    """
    error = np.abs(pred_disp - gt_disp)
    mae = np.mean(error)
    return mae

def calculate_bias(pred_disp, gt_disp):
    """
    Calculates Bias between predicted and ground truth displacements as:
    
    Bias = mean(pred_disp - gt_disp)
    
    Parameters:
    - pred_disp: 2D numpy array of predicted displacements.
    - gt_disp: 2D numpy array of ground truth displacements.
    
    Returns:
    - bias_mean: Mean Bias value.
    """
    bias = np.mean(pred_disp - gt_disp)
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

def calculate_rmse_general(pred_disp_x, pred_disp_y, gt_disp_x, gt_disp_y):
    """
    Calculates a general RMSE value considering both x and y displacements.
    
    Parameters:
    - pred_disp_x: 2D numpy array of predicted displacements in x-direction.
    - pred_disp_y: 2D numpy array of predicted displacements in y-direction.
    - gt_disp_x: 2D numpy array of ground truth displacements in x-direction.
    - gt_disp_y: 2D numpy array of ground truth displacements in y-direction.
    
    Returns:
    - rmse_general: General RMSE value.
    """
    # Flatten and concatenate both directions
    pred_all = np.concatenate((pred_disp_x.flatten(), pred_disp_y.flatten()))
    gt_all = np.concatenate((gt_disp_x.flatten(), gt_disp_y.flatten()))
    
    error = pred_all - gt_all
    mse = np.mean(error ** 2)
    rmse_general = np.sqrt(mse)
    return rmse_general

def calculate_bias_general(pred_disp_x, pred_disp_y, gt_disp_x, gt_disp_y):
    """
    Calculates a general Bias value considering both x and y displacements.
    
    Parameters:
    - pred_disp_x: 2D numpy array of predicted displacements in x-direction.
    - pred_disp_y: 2D numpy array of predicted displacements in y-direction.
    - gt_disp_x: 2D numpy array of ground truth displacements in x-direction.
    - gt_disp_y: 2D numpy array of ground truth displacements in y-direction.
    
    Returns:
    - bias_general_mean: Mean Bias value over both directions.
    """
    bias_x = calculate_bias(pred_disp_x, gt_disp_x)
    bias_y = calculate_bias(pred_disp_y, gt_disp_y)
    
    # General Bias is the average of bias_x and bias_y
    bias_general_mean = (bias_x + bias_y) / 2
    return bias_general_mean

def calculate_mae_general(pred_disp_x, pred_disp_y, gt_disp_x, gt_disp_y):
    """
    Calculates a general MAE value considering both x and y displacements.
    
    Parameters:
    - pred_disp_x: 2D numpy array of predicted displacements in x-direction.
    - pred_disp_y: 2D numpy array of predicted displacements in y-direction.
    - gt_disp_x: 2D numpy array of ground truth displacements in x-direction.
    - gt_disp_y: 2D numpy array of ground truth displacements in y-direction.
    
    Returns:
    - mae_general: General MAE value.
    """
    mae_x = calculate_mae(pred_disp_x, gt_disp_x)
    mae_y = calculate_mae(pred_disp_y, gt_disp_y)
    
    # General MAE is the average of mae_x and mae_y
    mae_general = (mae_x + mae_y) / 2
    return mae_general

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
        gpu = gpus[0]  # Select the first available GPU
        gpu_info = (f"GPU ID: {gpu.id}, Name: {gpu.name}, "
                    f"Temperature: {gpu.temperature} C, "
                    f"Memory Usage: {gpu.memoryUsed}/{gpu.memoryTotal} MB, "
                    f"Load: {gpu.load*100}%")
    else:
        gpu_info = "No GPU found."
    return gpu_info

def main(pred_dir, gt_dir, output_csv):
    """
    Calculates RMSE, MAE, Bias, and R^2 metrics for all displacement pairs.
    
    Parameters:
    - pred_dir: Directory containing predicted displacement files.
    - gt_dir: Directory containing ground truth displacement files.
    - output_csv: Path to the output CSV file for results.
    """
    # Display GPU information
    gpu_info = get_gpu_info()
    print(f"=== GPU Information ===\n{gpu_info}\n")
    
    # Find all predicted displacement x files
    pred_x_pattern = os.path.join(pred_dir, 'Def*_disp_x.csv')
    pred_x_files = sorted(glob.glob(pred_x_pattern))
    
    if not pred_x_files:
        print(f"No files found matching the pattern {pred_x_pattern}")
        return
    
    results = []
    
    # Use tqdm to show a progress bar
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
        rmse_general = calculate_rmse_general(disp_pred_x, disp_pred_y, disp_gt_x, disp_gt_y)
        
        mae_x = calculate_mae(disp_pred_x, disp_gt_x)
        mae_y = calculate_mae(disp_pred_y, disp_gt_y)
        mae_general = calculate_mae_general(disp_pred_x, disp_pred_y, disp_gt_x, disp_gt_y)
        
        bias_x = calculate_bias(disp_pred_x, disp_gt_x)
        bias_y = calculate_bias(disp_pred_y, disp_gt_y)
        bias_general = calculate_bias_general(disp_pred_x, disp_pred_y, disp_gt_x, disp_gt_y)
        
        r2_x = calculate_r2(disp_pred_x, disp_gt_x)
        r2_y = calculate_r2(disp_pred_y, disp_gt_y)
        r2_general = calculate_r2_general(disp_pred_x, disp_pred_y, disp_gt_x, disp_gt_y)
        
        # Store results
        results.append({
            'Identifier': identifier,
            'RMSE_x': rmse_x,
            'RMSE_y': rmse_y,
            'RMSE_general': rmse_general,
            'MAE_x': mae_x,
            'MAE_y': mae_y,
            'MAE_general': mae_general,
            'Bias_x': bias_x,
            'Bias_y': bias_y,
            'Bias_general': bias_general,
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
    summary = {
        'RMSE_x': (df_results['RMSE_x'].mean(), df_results['RMSE_x'].std()),
        'RMSE_y': (df_results['RMSE_y'].mean(), df_results['RMSE_y'].std()),
        'RMSE_general': (df_results['RMSE_general'].mean(), df_results['RMSE_general'].std()),
        'MAE_x': (df_results['MAE_x'].mean(), df_results['MAE_x'].std()),
        'MAE_y': (df_results['MAE_y'].mean(), df_results['MAE_y'].std()),
        'MAE_general': (df_results['MAE_general'].mean(), df_results['MAE_general'].std()),
        'Bias_x': (df_results['Bias_x'].mean(), df_results['Bias_x'].std()),
        'Bias_y': (df_results['Bias_y'].mean(), df_results['Bias_y'].std()),
        'Bias_general': (df_results['Bias_general'].mean(), df_results['Bias_general'].std()),
        'R2_x': (df_results['R2_x'].mean(), df_results['R2_x'].std()),
        'R2_y': (df_results['R2_y'].mean(), df_results['R2_y'].std()),
        'R2_general': (df_results['R2_general'].mean(), df_results['R2_general'].std())
    }
    
    # Display summary statistics
    print("=== RMSE Statistics ===")
    print(f"RMSE_x: Mean = {summary['RMSE_x'][0]:.6f}, Standard Deviation = {summary['RMSE_x'][1]:.6f}")
    print(f"RMSE_y: Mean = {summary['RMSE_y'][0]:.6f}, Standard Deviation = {summary['RMSE_y'][1]:.6f}")
    print(f"RMSE_general: Mean = {summary['RMSE_general'][0]:.6f}, Standard Deviation = {summary['RMSE_general'][1]:.6f}\n")
    
    print("=== MAE Statistics ===")
    print(f"MAE_x: Mean = {summary['MAE_x'][0]:.6f}, Standard Deviation = {summary['MAE_x'][1]:.6f}")
    print(f"MAE_y: Mean = {summary['MAE_y'][0]:.6f}, Standard Deviation = {summary['MAE_y'][1]:.6f}")
    print(f"MAE_general: Mean = {summary['MAE_general'][0]:.6f}, Standard Deviation = {summary['MAE_general'][1]:.6f}\n")
    
    print("=== Bias Statistics ===")
    print(f"Bias_x: Mean = {summary['Bias_x'][0]:.6f}, Standard Deviation = {summary['Bias_x'][1]:.6f}")
    print(f"Bias_y: Mean = {summary['Bias_y'][0]:.6f}, Standard Deviation = {summary['Bias_y'][1]:.6f}")
    print(f"Bias_general: Mean = {summary['Bias_general'][0]:.6f}, Standard Deviation = {summary['Bias_general'][1]:.6f}\n")
    
    print("=== R^2 Statistics ===")
    print(f"R2_x: Mean = {summary['R2_x'][0]:.6f}, Standard Deviation = {summary['R2_x'][1]:.6f}")
    print(f"R2_y: Mean = {summary['R2_y'][0]:.6f}, Standard Deviation = {summary['R2_y'][1]:.6f}")
    print(f"R2_general: Mean = {summary['R2_general'][0]:.6f}, Standard Deviation = {summary['R2_general'][1]:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate RMSE, MAE, Bias, and R^2 metrics for predicted displacements.')
    parser.add_argument('--pred_dir', type=str, default='/hpcfs/home/fisica/s.naranjob/StrainNet/StrainNet/output_inference/',
                        help='Directory containing predicted displacement files.')
    parser.add_argument('--gt_dir', type=str, default='/hpcfs/home/fisica/s.naranjob/StrainNet/Dataset/Test_Data/',
                        help='Directory containing ground truth displacement files.')
    parser.add_argument('--output_csv', type=str, default='metrics_results.csv',
                        help='CSV file to save the results.')
    
    args = parser.parse_args()
    
    main(args.pred_dir, args.gt_dir, args.output_csv)
