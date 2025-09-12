#!/usr/bin/env python3
"""
CSV Angle Interpolation Script
Processes CSV files in the datasets folder and fills in missing angle values through linear interpolation.
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path

def detect_valid_angle_rows(df):
    """
    Detect rows with valid angle data (9 values instead of 7).
    
    Args:
        df: DataFrame with CSV data
        
    Returns:
        List of row indices that contain valid angle data
    """
    valid_indices = []
    
    for idx, row in df.iterrows():
        # Check if row has 9 values (timestamp + angle1 + angle2 + 6 sensor values)
        if len(row) == 9:
            # Additional check: ensure angle values are not zero (indicating no angle data)
            if row['angle1'] != 0 or row['angle2'] != 0:
                valid_indices.append(idx)
    
    return valid_indices

def interpolate_angles_between_points(df, start_idx, end_idx):
    """
    Interpolate angle values between two known angle points.
    
    Args:
        df: DataFrame with CSV data
        start_idx: Index of first valid angle row
        end_idx: Index of last valid angle row
        
    Returns:
        DataFrame with interpolated angles
    """
    if start_idx >= end_idx:
        return df
    
    # Get the known angle values
    start_angles = df.iloc[start_idx][['angle1', 'angle2']].values
    end_angles = df.iloc[end_idx][['angle1', 'angle2']].values
    
    # Calculate number of rows to interpolate
    num_rows = end_idx - start_idx - 1
    
    if num_rows <= 0:
        return df
    
    # Create interpolation arrays
    angle1_interp = np.linspace(start_angles[0], end_angles[0], num_rows + 2)[1:-1]
    angle2_interp = np.linspace(start_angles[1], end_angles[1], num_rows + 2)[1:-1]
    
    # Fill in the interpolated values
    for i, (angle1, angle2) in enumerate(zip(angle1_interp, angle2_interp)):
        row_idx = start_idx + 1 + i
        df.at[row_idx, 'angle1'] = angle1
        df.at[row_idx, 'angle2'] = angle2
    
    return df

def process_csv_file(file_path):
    """
    Process a single CSV file and interpolate missing angle values.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with interpolated angles
    """
    print(f"Processing: {file_path}")
    
    # Read CSV file
    df = pd.read_csv(file_path)
    
    # Ensure we have the expected columns for the actual data format
    expected_columns = ['timestamp', 'angle1', 'angle2', 'A11', 'D11', 'D10', 'D9', 'D8', 'D7']
    if not all(col in df.columns for col in expected_columns):
        print(f"Warning: {file_path} does not have expected columns. Expected: {expected_columns}")
        print(f"Actual columns: {list(df.columns)}")
        return None
    
    # Detect rows with valid angle data
    valid_indices = detect_valid_angle_rows(df)
    
    if len(valid_indices) < 2:
        print(f"Warning: {file_path} has fewer than 2 valid angle rows. Skipping.")
        return None
    
    print(f"Found {len(valid_indices)} valid angle rows")
    
    # Handle leading rows before first valid angle (remove them)
    first_valid_idx = valid_indices[0]
    leading_rows = first_valid_idx
    
    if leading_rows > 0:
        print(f"  Removing {leading_rows} leading rows before first valid angle at index {first_valid_idx}")
        df = df.iloc[first_valid_idx:].copy()  # Remove everything before first valid row
        # Adjust valid indices to account for removed rows
        valid_indices = [idx - first_valid_idx for idx in valid_indices]
    
    # Process each sequence between valid angle points
    for i in range(len(valid_indices) - 1):
        start_idx = valid_indices[i]
        end_idx = valid_indices[i + 1]
        
        # Interpolate angles between these two points
        df = interpolate_angles_between_points(df, start_idx, end_idx)
        
        print(f"  Interpolated {end_idx - start_idx - 1} rows between indices {start_idx} and {end_idx}")
    
    # Handle trailing rows that can't be interpolated
    last_valid_idx = valid_indices[-1]
    trailing_rows = len(df) - last_valid_idx - 1
    
    if trailing_rows > 0:
        print(f"  Removing {trailing_rows} trailing rows after last valid angle at index {last_valid_idx}")
        df = df.iloc[:last_valid_idx + 1].copy()  # Keep the last valid row, remove everything after
    
    return df

def process_all_csv_files(datasets_dir="datasets"):
    """
    Process all CSV files in the datasets directory.
    
    Args:
        datasets_dir: Directory containing CSV files
    """
    # Find all CSV files
    csv_files = glob.glob(os.path.join(datasets_dir, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {datasets_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    print("=" * 50)
    
    # Create output directory for interpolated files
    output_dir = os.path.join(datasets_dir, "interpolated")
    os.makedirs(output_dir, exist_ok=True)
    
    processed_count = 0
    
    for csv_file in sorted(csv_files):
        try:
            # Process the file
            interpolated_df = process_csv_file(csv_file)
            
            if interpolated_df is not None:
                # Save interpolated file
                filename = os.path.basename(csv_file)
                output_path = os.path.join(output_dir, f"interpolated_{filename}")
                interpolated_df.to_csv(output_path, index=False)
                
                print(f"Saved interpolated file: {output_path}")
                processed_count += 1
            else:
                print(f"Skipped: {csv_file}")
            
            print("-" * 30)
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            print("-" * 30)
    
    print(f"Processing complete! {processed_count} files processed successfully.")
    print(f"Interpolated files saved in: {output_dir}")

def analyze_interpolation_quality(datasets_dir="datasets"):
    """
    Analyze the quality of interpolation by comparing original and interpolated files.
    """
    original_dir = datasets_dir
    interpolated_dir = os.path.join(datasets_dir, "interpolated")
    
    if not os.path.exists(interpolated_dir):
        print("No interpolated files found. Run interpolation first.")
        return
    
    print("Analyzing interpolation quality...")
    print("=" * 50)
    
    # Find corresponding files
    original_files = glob.glob(os.path.join(original_dir, "*.csv"))
    
    for original_file in sorted(original_files):
        filename = os.path.basename(original_file)
        interpolated_file = os.path.join(interpolated_dir, f"interpolated_{filename}")
        
        if os.path.exists(interpolated_file):
            try:
                # Load both files
                original_df = pd.read_csv(original_file)
                interpolated_df = pd.read_csv(interpolated_file)
                
                # Count original valid angle rows
                original_valid = len(detect_valid_angle_rows(original_df))
                
                # Count interpolated angle rows (non-zero angles)
                interpolated_valid = len(interpolated_df[(interpolated_df['angle1'] != 0) | (interpolated_df['angle2'] != 0)])
                
                print(f"{filename}:")
                print(f"  Original valid angle rows: {original_valid}")
                print(f"  Total rows with angles after interpolation: {interpolated_valid}")
                print(f"  Interpolated rows: {interpolated_valid - original_valid}")
                print(f"  Final file length: {len(interpolated_df)} rows")
                
                if interpolated_valid > original_valid:
                    # Calculate some statistics
                    angle1_stats = interpolated_df['angle1'].describe()
                    angle2_stats = interpolated_df['angle2'].describe()
                    
                    print(f"  Angle1 range: {angle1_stats['min']:.1f} - {angle1_stats['max']:.1f}")
                    print(f"  Angle2 range: {angle2_stats['min']:.1f} - {angle2_stats['max']:.1f}")
                
                print()
                
            except Exception as e:
                print(f"Error analyzing {filename}: {e}")

def main():
    """Main function."""
    print("CSV Angle Interpolation Script")
    print("=" * 50)
    print("This script processes CSV files and interpolates missing angle values.")
    print("Expected format: timestamp, angle1, angle2, A11, D11, D10, D9, D8, D7")
    print()
    
    # Process all CSV files
    process_all_csv_files()
    
    print()
    
    # Analyze interpolation quality
    analyze_interpolation_quality()
    
    print()
    print("Interpolation complete!")
    print("Check the 'datasets/interpolated/' directory for processed files.")

if __name__ == "__main__":
    main() 