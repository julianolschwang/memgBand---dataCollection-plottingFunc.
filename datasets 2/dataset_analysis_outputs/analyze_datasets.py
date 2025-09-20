#!/usr/bin/env python3
"""
Dataset Analysis Script for v5.12.11
====================================

Comprehensive analysis of all datasets in the datasets folder to understand:
- Global angle1 and angle2 statistics
- Data distribution patterns
- Skewness and bias detection
- Dataset comparison and quality assessment
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats as scipy_stats
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class DatasetAnalyzer:
    """Comprehensive dataset analysis for EMG-angle prediction data"""
    
    def __init__(self, datasets_folder="datasets"):
        self.datasets_folder = Path(datasets_folder)
        self.results = {}
        self.global_stats = {}
        
    def find_all_datasets(self):
        """Find all CSV files in the datasets folder"""
        print("🔍 Scanning for datasets...")
        
        # Find all CSV files in both og and augmented folders
        og_files = list(self.datasets_folder.glob("og/*.csv"))
        augmented_files = list(self.datasets_folder.glob("augmented/*.csv"))
        
        all_files = og_files + augmented_files
        
        print(f"   📁 Found {len(og_files)} original datasets")
        print(f"   📁 Found {len(augmented_files)} augmented datasets")
        print(f"   📊 Total datasets: {len(all_files)}")
        
        return all_files
    
    def analyze_single_dataset(self, file_path):
        """Analyze a single dataset file"""
        try:
            # Load dataset
            df = pd.read_csv(file_path)
            
            # Identify angle columns
            angle_columns = [col for col in df.columns if col.startswith('angle')]
            
            if len(angle_columns) < 2:
                print(f"   ⚠️  {file_path.name}: Only {len(angle_columns)} angle columns found")
                return None
            
            # Convert to numeric
            for col in angle_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove NaN values
            df = df.dropna(subset=angle_columns)
            
            if len(df) == 0:
                print(f"   ⚠️  {file_path.name}: No valid data after cleaning")
                return None
            
            # Extract angle data
            angle1 = df[angle_columns[0]].values
            angle2 = df[angle_columns[1]].values if len(angle_columns) > 1 else None
            
            # Calculate statistics
            stats_dict = {
                'file_name': file_path.name,
                'file_path': str(file_path),
                'total_samples': len(df),
                'angle1': self._calculate_angle_stats(angle1, 'angle1'),
                'angle2': self._calculate_angle_stats(angle2, 'angle2') if angle2 is not None else None,
                'dataset_type': 'original' if 'og' in str(file_path) else 'augmented'
            }
            
            return stats_dict
            
        except Exception as e:
            print(f"   ❌ Error analyzing {file_path.name}: {str(e)}")
            return None
    
    def _calculate_angle_stats(self, angle_data, angle_name):
        """Calculate comprehensive statistics for angle data"""
        if angle_data is None or len(angle_data) == 0:
            return None
        
        # Basic statistics
        mean_val = np.mean(angle_data)
        std_val = np.std(angle_data)
        min_val = np.min(angle_data)
        max_val = np.max(angle_data)
        median_val = np.median(angle_data)
        
        # Distribution statistics
        skewness = scipy_stats.skew(angle_data)
        kurtosis = scipy_stats.kurtosis(angle_data)
        
        # Percentiles
        percentiles = {
            'p25': np.percentile(angle_data, 25),
            'p75': np.percentile(angle_data, 75),
            'p90': np.percentile(angle_data, 90),
            'p95': np.percentile(angle_data, 95),
            'p99': np.percentile(angle_data, 99)
        }
        
        # Range analysis
        range_val = max_val - min_val
        iqr = percentiles['p75'] - percentiles['p25']
        
        # Outlier detection (using IQR method)
        q1, q3 = percentiles['p25'], percentiles['p75']
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = np.sum((angle_data < lower_bound) | (angle_data > upper_bound))
        outlier_percentage = (outliers / len(angle_data)) * 100
        
        # Distribution shape analysis
        if abs(skewness) < 0.5:
            skewness_interpretation = "approximately normal"
        elif abs(skewness) < 1:
            skewness_interpretation = "moderately skewed"
        else:
            skewness_interpretation = "highly skewed"
        
        if abs(kurtosis) < 0.5:
            kurtosis_interpretation = "mesokurtic (normal-like)"
        elif kurtosis > 0.5:
            kurtosis_interpretation = "leptokurtic (heavy-tailed)"
        else:
            kurtosis_interpretation = "platykurtic (light-tailed)"
        
        return {
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'median': median_val,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'range': range_val,
            'iqr': iqr,
            'outliers_count': outliers,
            'outliers_percentage': outlier_percentage,
            'skewness_interpretation': skewness_interpretation,
            'kurtosis_interpretation': kurtosis_interpretation,
            'percentiles': percentiles
        }
    
    def calculate_global_statistics(self):
        """Calculate global statistics across all datasets"""
        print("\n📊 Calculating Global Statistics...")
        
        # Collect all angle data
        all_angle1 = []
        all_angle2 = []
        dataset_sizes = []
        
        for file_name, stats in self.results.items():
            if stats['angle1'] is not None:
                # Load the actual data for global analysis
                df = pd.read_csv(stats['file_path'])
                angle_columns = [col for col in df.columns if col.startswith('angle')]
                
                if len(angle_columns) >= 2:
                    for col in angle_columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    df = df.dropna(subset=angle_columns)
                    
                    all_angle1.extend(df[angle_columns[0]].values)
                    all_angle2.extend(df[angle_columns[1]].values)
                    dataset_sizes.append(len(df))
        
        # Convert to numpy arrays
        all_angle1 = np.array(all_angle1)
        all_angle2 = np.array(all_angle2)
        
        print(f"   📈 Total samples analyzed: {len(all_angle1):,}")
        print(f"   📊 Average dataset size: {np.mean(dataset_sizes):.0f} samples")
        print(f"   📊 Dataset size range: {np.min(dataset_sizes):,} - {np.max(dataset_sizes):,} samples")
        
        # Calculate global statistics
        self.global_stats = {
            'total_samples': len(all_angle1),
            'num_datasets': len(self.results),
            'angle1': self._calculate_angle_stats(all_angle1, 'angle1'),
            'angle2': self._calculate_angle_stats(all_angle2, 'angle2'),
            'dataset_sizes': {
                'mean': np.mean(dataset_sizes),
                'std': np.std(dataset_sizes),
                'min': np.min(dataset_sizes),
                'max': np.max(dataset_sizes),
                'median': np.median(dataset_sizes)
            }
        }
        
        return self.global_stats
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations"""
        print("\n📊 Generating Visualizations...")
        
        # Create output directory
        output_dir = Path("dataset_analysis_outputs")
        output_dir.mkdir(exist_ok=True)
        
        # 1. Global Distribution Plots
        self._plot_global_distributions(output_dir)
        
        # 2. Dataset Comparison Plots
        self._plot_dataset_comparisons(output_dir)
        
        # 3. Statistical Summary Plots
        self._plot_statistical_summaries(output_dir)
        
        print(f"   📁 Visualizations saved to: {output_dir}")
        
    def _plot_global_distributions(self, output_dir):
        """Plot global angle distributions"""
        # Collect all angle data
        all_angle1 = []
        all_angle2 = []
        
        for file_name, stats in self.results.items():
            if stats['angle1'] is not None:
                df = pd.read_csv(stats['file_path'])
                angle_columns = [col for col in df.columns if col.startswith('angle')]
                
                if len(angle_columns) >= 2:
                    for col in angle_columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    df = df.dropna(subset=angle_columns)
                    
                    all_angle1.extend(df[angle_columns[0]].values)
                    all_angle2.extend(df[angle_columns[1]].values)
        
        all_angle1 = np.array(all_angle1)
        all_angle2 = np.array(all_angle2)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Global Angle Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Angle1 plots
        axes[0, 0].hist(all_angle1, bins=100, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Angle1 - Global Distribution')
        axes[0, 0].set_xlabel('Angle1 Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].boxplot(all_angle1, patch_artist=True)
        axes[0, 1].set_title('Angle1 - Box Plot')
        axes[0, 1].set_ylabel('Angle1 Value')
        axes[0, 1].grid(True, alpha=0.3)
        
        scipy_stats.probplot(all_angle1, dist="norm", plot=axes[0, 2])
        axes[0, 2].set_title('Angle1 - Q-Q Plot (Normal Distribution)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Angle2 plots
        axes[1, 0].hist(all_angle2, bins=100, alpha=0.7, edgecolor='black', color='orange')
        axes[1, 0].set_title('Angle2 - Global Distribution')
        axes[1, 0].set_xlabel('Angle2 Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].boxplot(all_angle2, patch_artist=True)
        axes[1, 1].set_title('Angle2 - Box Plot')
        axes[1, 1].set_ylabel('Angle2 Value')
        axes[1, 1].grid(True, alpha=0.3)
        
        scipy_stats.probplot(all_angle2, dist="norm", plot=axes[1, 2])
        axes[1, 2].set_title('Angle2 - Q-Q Plot (Normal Distribution)')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'global_angle_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Scatter plot of angle1 vs angle2
        plt.figure(figsize=(10, 8))
        plt.scatter(all_angle1, all_angle2, alpha=0.5, s=1)
        plt.xlabel('Angle1')
        plt.ylabel('Angle2')
        plt.title('Angle1 vs Angle2 - Global Scatter Plot')
        plt.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        correlation = np.corrcoef(all_angle1, all_angle2)[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {correlation:.4f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_dir / 'angle_correlation.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_dataset_comparisons(self, output_dir):
        """Plot comparisons between datasets"""
        # Prepare data for comparison
        dataset_names = []
        angle1_means = []
        angle2_means = []
        angle1_stds = []
        angle2_stds = []
        dataset_sizes = []
        
        for file_name, stats in self.results.items():
            if stats['angle1'] is not None and stats['angle2'] is not None:
                dataset_names.append(file_name)
                angle1_means.append(stats['angle1']['mean'])
                angle2_means.append(stats['angle2']['mean'])
                angle1_stds.append(stats['angle1']['std'])
                angle2_stds.append(stats['angle2']['std'])
                dataset_sizes.append(stats['total_samples'])
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Dataset Comparison Analysis', fontsize=16, fontweight='bold')
        
        # Mean values comparison
        x_pos = np.arange(len(dataset_names))
        width = 0.35
        
        axes[0, 0].bar(x_pos - width/2, angle1_means, width, label='Angle1', alpha=0.7)
        axes[0, 0].bar(x_pos + width/2, angle2_means, width, label='Angle2', alpha=0.7)
        axes[0, 0].set_xlabel('Dataset')
        axes[0, 0].set_ylabel('Mean Value')
        axes[0, 0].set_title('Mean Angle Values by Dataset')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Standard deviation comparison
        axes[0, 1].bar(x_pos - width/2, angle1_stds, width, label='Angle1', alpha=0.7)
        axes[0, 1].bar(x_pos + width/2, angle2_stds, width, label='Angle2', alpha=0.7)
        axes[0, 1].set_xlabel('Dataset')
        axes[0, 1].set_ylabel('Standard Deviation')
        axes[0, 1].set_title('Angle Variability by Dataset')
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Dataset sizes
        axes[1, 0].bar(x_pos, dataset_sizes, alpha=0.7, color='green')
        axes[1, 0].set_xlabel('Dataset')
        axes[1, 0].set_ylabel('Number of Samples')
        axes[1, 0].set_title('Dataset Sizes')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Skewness comparison
        angle1_skewness = [self.results[name]['angle1']['skewness'] for name in dataset_names]
        angle2_skewness = [self.results[name]['angle2']['skewness'] for name in dataset_names]
        
        axes[1, 1].bar(x_pos - width/2, angle1_skewness, width, label='Angle1', alpha=0.7)
        axes[1, 1].bar(x_pos + width/2, angle2_skewness, width, label='Angle2', alpha=0.7)
        axes[1, 1].set_xlabel('Dataset')
        axes[1, 1].set_ylabel('Skewness')
        axes[1, 1].set_title('Angle Skewness by Dataset')
        axes[1, 1].legend()
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'dataset_comparisons.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_statistical_summaries(self, output_dir):
        """Plot statistical summary visualizations"""
        # Create summary statistics table
        summary_data = []
        
        for file_name, stats in self.results.items():
            if stats['angle1'] is not None and stats['angle2'] is not None:
                summary_data.append({
                    'Dataset': file_name,
                    'Samples': stats['total_samples'],
                    'Angle1_Mean': stats['angle1']['mean'],
                    'Angle1_Std': stats['angle1']['std'],
                    'Angle1_Skew': stats['angle1']['skewness'],
                    'Angle2_Mean': stats['angle2']['mean'],
                    'Angle2_Std': stats['angle2']['std'],
                    'Angle2_Skew': stats['angle2']['skewness'],
                    'Type': stats['dataset_type']
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Create heatmap of statistics
        plt.figure(figsize=(12, 8))
        
        # Select numeric columns for heatmap
        heatmap_data = summary_df[['Samples', 'Angle1_Mean', 'Angle1_Std', 'Angle1_Skew', 
                                 'Angle2_Mean', 'Angle2_Std', 'Angle2_Skew']].T
        heatmap_data.columns = summary_df['Dataset']
        
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlBu_r', center=0)
        plt.title('Dataset Statistics Heatmap')
        plt.xlabel('Dataset')
        plt.ylabel('Statistic')
        plt.tight_layout()
        plt.savefig(output_dir / 'statistics_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("\n📋 Generating Analysis Report...")
        
        # Create output directory
        output_dir = Path("dataset_analysis_outputs")
        output_dir.mkdir(exist_ok=True)
        
        report_path = output_dir / "dataset_analysis_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DATASET ANALYSIS REPORT - v5.12.11\n")
            f.write("=" * 80 + "\n\n")
            
            # Global statistics
            f.write("GLOBAL STATISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total datasets analyzed: {self.global_stats['num_datasets']}\n")
            f.write(f"Total samples: {self.global_stats['total_samples']:,}\n")
            f.write(f"Average dataset size: {self.global_stats['dataset_sizes']['mean']:.0f} samples\n")
            f.write(f"Dataset size range: {self.global_stats['dataset_sizes']['min']:,} - {self.global_stats['dataset_sizes']['max']:,} samples\n\n")
            
            # Angle1 global statistics
            if self.global_stats['angle1']:
                f.write("ANGLE1 GLOBAL STATISTICS\n")
                f.write("-" * 25 + "\n")
                angle1_stats = self.global_stats['angle1']
                f.write(f"Mean: {angle1_stats['mean']:.6f}\n")
                f.write(f"Standard Deviation: {angle1_stats['std']:.6f}\n")
                f.write(f"Range: {angle1_stats['min']:.6f} to {angle1_stats['max']:.6f}\n")
                f.write(f"Median: {angle1_stats['median']:.6f}\n")
                f.write(f"Skewness: {angle1_stats['skewness']:.6f} ({angle1_stats['skewness_interpretation']})\n")
                f.write(f"Kurtosis: {angle1_stats['kurtosis']:.6f} ({angle1_stats['kurtosis_interpretation']})\n")
                f.write(f"Outliers: {angle1_stats['outliers_count']:,} ({angle1_stats['outliers_percentage']:.2f}%)\n")
                f.write(f"IQR: {angle1_stats['iqr']:.6f}\n\n")
            
            # Angle2 global statistics
            if self.global_stats['angle2']:
                f.write("ANGLE2 GLOBAL STATISTICS\n")
                f.write("-" * 25 + "\n")
                angle2_stats = self.global_stats['angle2']
                f.write(f"Mean: {angle2_stats['mean']:.6f}\n")
                f.write(f"Standard Deviation: {angle2_stats['std']:.6f}\n")
                f.write(f"Range: {angle2_stats['min']:.6f} to {angle2_stats['max']:.6f}\n")
                f.write(f"Median: {angle2_stats['median']:.6f}\n")
                f.write(f"Skewness: {angle2_stats['skewness']:.6f} ({angle2_stats['skewness_interpretation']})\n")
                f.write(f"Kurtosis: {angle2_stats['kurtosis']:.6f} ({angle2_stats['kurtosis_interpretation']})\n")
                f.write(f"Outliers: {angle2_stats['outliers_count']:,} ({angle2_stats['outliers_percentage']:.2f}%)\n")
                f.write(f"IQR: {angle2_stats['iqr']:.6f}\n\n")
            
            # Dataset-by-dataset analysis
            f.write("DATASET-BY-DATASET ANALYSIS\n")
            f.write("-" * 30 + "\n")
            
            for file_name, stats in self.results.items():
                if stats['angle1'] is not None and stats['angle2'] is not None:
                    f.write(f"\nDataset: {file_name}\n")
                    f.write(f"Type: {stats['dataset_type']}\n")
                    f.write(f"Samples: {stats['total_samples']:,}\n")
                    
                    f.write(f"  Angle1 - Mean: {stats['angle1']['mean']:.6f}, Std: {stats['angle1']['std']:.6f}, Skew: {stats['angle1']['skewness']:.6f}\n")
                    f.write(f"  Angle2 - Mean: {stats['angle2']['mean']:.6f}, Std: {stats['angle2']['std']:.6f}, Skew: {stats['angle2']['skewness']:.6f}\n")
            
            # Recommendations
            f.write("\n\nRECOMMENDATIONS\n")
            f.write("-" * 15 + "\n")
            
            if self.global_stats['angle1'] and self.global_stats['angle2']:
                angle1_skew = abs(self.global_stats['angle1']['skewness'])
                angle2_skew = abs(self.global_stats['angle2']['skewness'])
                
                if angle1_skew > 1 or angle2_skew > 1:
                    f.write("⚠️  HIGH SKEWNESS DETECTED:\n")
                    f.write("   - Consider data transformation (log, sqrt, Box-Cox)\n")
                    f.write("   - Check for outliers that might be causing skewness\n")
                    f.write("   - Consider robust normalization methods\n\n")
                
                if self.global_stats['angle1']['outliers_percentage'] > 5 or self.global_stats['angle2']['outliers_percentage'] > 5:
                    f.write("⚠️  HIGH OUTLIER PERCENTAGE:\n")
                    f.write("   - Review outlier detection and handling strategy\n")
                    f.write("   - Consider robust statistics for model training\n\n")
                
                f.write("✅ DATA QUALITY ASSESSMENT:\n")
                f.write(f"   - Angle1 skewness: {self.global_stats['angle1']['skewness']:.3f} ({self.global_stats['angle1']['skewness_interpretation']})\n")
                f.write(f"   - Angle2 skewness: {self.global_stats['angle2']['skewness']:.3f} ({self.global_stats['angle2']['skewness_interpretation']})\n")
                f.write(f"   - Total outliers: {self.global_stats['angle1']['outliers_count'] + self.global_stats['angle2']['outliers_count']:,}\n")
                f.write(f"   - Data coverage: {self.global_stats['total_samples']:,} samples across {self.global_stats['num_datasets']} datasets\n")
        
        print(f"   📄 Report saved: {report_path}")
    
    def run_analysis(self):
        """Run complete dataset analysis"""
        print("🚀 Starting Comprehensive Dataset Analysis")
        print("=" * 50)
        
        # Find all datasets
        dataset_files = self.find_all_datasets()
        
        if not dataset_files:
            print("❌ No datasets found!")
            return
        
        # Analyze each dataset
        print(f"\n📊 Analyzing {len(dataset_files)} datasets...")
        for i, file_path in enumerate(dataset_files, 1):
            print(f"   [{i:3d}/{len(dataset_files)}] {file_path.name}", end=" ... ")
            result = self.analyze_single_dataset(file_path)
            if result:
                self.results[file_path.name] = result
                print("✅")
            else:
                print("❌")
        
        print(f"\n✅ Successfully analyzed {len(self.results)} datasets")
        
        # Calculate global statistics
        self.calculate_global_statistics()
        
        # Generate visualizations
        self.generate_visualizations()
        
        # Generate report
        self.generate_report()
        
        print(f"\n🎉 Analysis Complete!")
        print(f"📁 Results saved to: dataset_analysis_outputs/")
        
        return self.results, self.global_stats

def main():
    """Main execution function"""
    analyzer = DatasetAnalyzer()
    results, global_stats = analyzer.run_analysis()
    
    # Print summary
    print("\n" + "="*60)
    print("QUICK SUMMARY")
    print("="*60)
    print(f"📊 Datasets analyzed: {global_stats['num_datasets']}")
    print(f"📈 Total samples: {global_stats['total_samples']:,}")
    
    if global_stats['angle1']:
        print(f"📐 Angle1 - Mean: {global_stats['angle1']['mean']:.3f}, Skew: {global_stats['angle1']['skewness']:.3f}")
    if global_stats['angle2']:
        print(f"📐 Angle2 - Mean: {global_stats['angle2']['mean']:.3f}, Skew: {global_stats['angle2']['skewness']:.3f}")

if __name__ == "__main__":
    main()
