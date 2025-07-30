#!/usr/bin/env python3
# Original Code by Vangara Yaswanth Sai 
# Built for Flipkart Task-1 Robust Data Pipeline for Sales Insights 
# Task Given :-  Build a robust data pipeline that consolidates sales data from multiple formats (CSV, JSON, Excel), 
#                cleans it, performs transformations, and delivers key business insights through reports and visualizations.
# Date(Last Updated): 15-07-2025   
"""
Custom Multi-Source Sales Data Pipeline & Business Insights Dashboard
Modified to work with your existing data files

This project demonstrates:
1. Multi-format data ingestion (CSV, JSON, Excel) - YOUR FILES
2. Data cleaning and transformation
3. Data merging and integration
4. Business insights generation
5. Visualization and reporting

Required Libraries: pandas, numpy, matplotlib, seaborn, tabulate, openpyxl, json
"""
#TODO: Making Automation for User Comfort ( Make the code to be able to handle Large Datasets and Big Data - Take some from Kaggle )

# Importing the Required Libraries
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from datetime import datetime,timedelta
import warnings
import os
from pathlib import Path
import time # For using this a single time in whole code ( Maybe used for Automation )

# Configure display settings for the device 
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Here starts the root of the tree or Basin to the pipes about to be pulled
class RobustSalesDataPipeline:
    """
    A comprehensive sales data pipeline that handles YOUR data files,
    performs data cleaning, transformation, and generates business insights.
    """
    # This one calls himself as no one else would
    def __init__(self, csv_file=None, json_file=None, excel_file=None):
        self.csv_file =  r"c:\Users\yaswa\OneDrive\Desktop\Flipkart Project\multisource_sales_dashboard_demo\users_data.csv" # Add CSV path
        self.json_file = r"c:\Users\yaswa\OneDrive\Desktop\Flipkart Project\multisource_sales_dashboard_demo\mcc_codes.json" # Add JSON path
        self.excel_file = r"c:\Users\yaswa\OneDrive\Desktop\Flipkart Project\multisource_sales_dashboard_demo\region_info.xlsx" # Add Excel path
        self.users_data = None
        self.metadata = None
        self.region_data = None
        self.merged_data = None
        self.insights = {}

    def inspect_data_files(self):
        """Inspect your data files to understand their structure"""
        print("\nINSPECTING YOUR DATA FILES")
        print("=" * 50)
        # Inspect CSV file
        if self.csv_file and os.path.exists(self.csv_file):
            print(f"\nCSV File: {self.csv_file}")
            print("-" * 30)
            try:
                df_sample = pd.read_csv(self.csv_file, nrows=5)
                print("Columns:", list(df_sample.columns))
                print("Shape:", df_sample.shape)
                print("Sample data:")
                print(df_sample.head())
                print("\nData types:")
                print(df_sample.dtypes)
            except (FileNotFoundError, pd.errors.ParserError, OSError) as e:
                print(f"Error reading the CSV file - Reason: {e}")
        # Inspect JSON file
        if self.json_file and os.path.exists(self.json_file):
            print(f"\nJSON File: {self.json_file}")
            print("-" * 30)
            try:
                with open(self.json_file, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    print("Type: List of objects")
                    print("Number of records:", len(data))
                    print("Sample record keys:", list(data[0].keys()) if data else "Empty Record keys")
                    print("Sample record:")
                    print(json.dumps(data[0] if data else {}, indent=2))
                elif isinstance(data, dict):
                    print("Type: Dictionary")
                    print("Keys:", list(data.keys()))
                    print("Sample structure:")
                    print(json.dumps(data, indent=2)[:500] + "..." if len(str(data)) > 500 else json.dumps(data, indent=2))
            except (FileNotFoundError, pd.errors.ParserError, OSError) as e:
                print(f"Error reading the JSON file - Reason: {e}")
        # Inspect Excel file
        if self.excel_file and os.path.exists(self.excel_file):
            print(f"\nExcel File: {self.excel_file}")
            print("-" * 30)
            try:
                # Check available sheets
                excel_file = pd.ExcelFile(self.excel_file)
                print("Available sheets:", excel_file.sheet_names)
                # Read first sheet sample
                df_sample = pd.read_excel(self.excel_file, nrows=5)
                print("Columns:", list(df_sample.columns))
                print("Shape:", df_sample.shape)
                print("Sample data:")
                print(df_sample.head())
                print("\nData types:")
                print(df_sample.dtypes)
            except (FileNotFoundError, ValueError, OSError) as e:
                print(f"Error reading the Excel file - Reason: {e}")
        print("Data Inspection completed !!!")
    # Loading the CSV data into Pipeline
    def load_csv_file_data(self, date_columns=None, numeric_columns=None):
        """Load and validate CSV data with flexible column detection
           Tip: If dates break, try date_columns=['order_date']
        """
        if not self.csv_file or not os.path.exists(self.csv_file):
            print(f"CSV file not found: {self.csv_file}")
            return None
        try:
            print(f"Loading CSV data from {self.csv_file}...")
            data = pd.read_csv(self.csv_file)
            print(f"Loaded {len(data)} records with {len(data.columns)} columns")
            # Auto-detect date columns if not specified
            if date_columns is None:
                date_columns = []
                for col in data.columns:
                    if any(keyword in col.lower() for keyword in ['date', 'time', 'created', 'updated']):
                        date_columns.append(col)
            # Convert date columns
            for col in date_columns:
                if col in data.columns:
                    try:
                        data[col] = pd.to_datetime(data[col])
                        print(f"Converted {col} to datetime")
                    except:
                        print(f"Could not convert {col} to datetime")
            # Auto-detect numeric columns if not specified
            if numeric_columns is None:
                numeric_columns = []
                for col in data.columns:
                    if any(keyword in col.lower() for keyword in ['price', 'cost', 'revenue', 'amount', 'quantity', 'units', 'sales']):
                        numeric_columns.append(col)
            # Convert numeric columns
            for col in numeric_columns:
                if col in data.columns:
                    try:
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                        print(f"Converted {col} to numeric")
                    except:
                        print(f"Could not convert {col} to numeric")
            # Remove rows with all NaN values
            data = data.dropna(how='all')
            print(f"Cleaned data: {len(data)} records")
            return data  
        except (FileNotFoundError, pd.errors.ParserError, ValueError, OSError) as e:
            print(f" CSV Loading has been Failed - Probable cause: {e}")
            return None
    #Loading the JSON data into Pipeline
    def load_json_file_data(self):
        """Load and validate JSON metadata with flexible structure"""
        if not self.json_file or not os.path.exists(self.json_file):
            print(f"JSON file not found: {self.json_file}")
            return None
        try:
            print(f"Loading JSON data from {self.json_file}...")
            with open(self.json_file, 'r') as f:
                data = json.load(f)
            # Handle different JSON structures
            if isinstance(data, list):
                sales_Metadata = pd.DataFrame(data)
            elif isinstance(data, dict):
                # Try to find the main data array
                if 'data' in data:
                    sales_Metadata = pd.DataFrame(data['data'])
                elif 'products' in data:
                    sales_Metadata = pd.DataFrame(data['products'])
                elif 'items' in data:
                    sales_Metadata = pd.DataFrame(data['items'])
                else:
                    # Convert dict to single-row DataFrame
                    sales_Metadata = pd.DataFrame([data])
            else:
                print("Unsupported JSON structure - Please check again")
                return None 
            print(f"Loaded {len(sales_Metadata)} records from JSON")
            return sales_Metadata 
        except (FileNotFoundError, json.JSONDecodeError, TypeError, OSError) as e:
            print(f" JSON Loading has been Failed - Probable cause: {e}")
            return None
    # Loading the Excel into Pipeline
    def load_excel_file_data(self, sheet_name=0):
        """Load and validate Excel data with flexible sheet selection"""
        if not self.excel_file or not os.path.exists(self.excel_file):
            print(f"Excel file not found: {self.excel_file}")
            return None
        try:
            print(f"Loading Excel data from {self.excel_file}...")  
            # Read the specified sheet
            data = pd.read_excel(self.excel_file, sheet_name=sheet_name)  
            # Clean column names (remove spaces, special characters)
            data.columns = data.columns.str.strip() 
            print(f"Loaded {len(data)} records from Excel")
            return data  
        except (FileNotFoundError, ValueError, OSError) as e:
            print(f"Excel Loading has been Failed - Probable cause: {e}")
            return None
    # Data Merging process of Multiple formats ( if existed )
    def sales_merge_data(self, merge_keys=None):
        """Merge the loaded data sources based on common columns"""
        print("\n" + "="*50)
        print("SALES DATA MERGING BEGINS...")
        print("="*50)
        if self.users_data is None:
            print("No primary data available for merging")
            return None
        merged = self.users_data.copy()
        print(f"Starting with primary data: {len(merged)} records")
        # Sales merge with metadata
        if self.metadata is not None:
            print("\nMerging with metadata...")
            # Find common columns for merging
            if merge_keys is None:
                common_cols = list(set(merged.columns) & set(self.metadata.columns))
                if common_cols:
                    merge_key = common_cols[0]
                    print(f"Auto-detected merge key: {merge_key}")
                else:
                    print("No common columns found for merging metadata")
                    merge_key = None
            else:
                merge_key = merge_keys.get('metadata')
            if merge_key and merge_key in merged.columns and merge_key in self.metadata.columns:
                merged = pd.merge(merged, self.metadata, on=merge_key, how='left', suffixes=('', '_meta'))
                print(f"Merged with metadata: {len(merged)} records")
            else:
                print("Skipping metadata merge - no suitable key found to merge")
        # Sales merge with region/additional data ( Mostly not useful, just for the fluid Datasets )
        if self.region_data is not None:
            print("\nMerging with additional data...")
            if merge_keys is None:
                common_cols = list(set(merged.columns) & set(self.region_data.columns))
                if common_cols:
                    merge_key = common_cols[0]
                    print(f"Auto-detected merge key: {merge_key}")
                else:
                    print("No common columns found for merging additional data")
                    merge_key = None
            else:
                merge_key = merge_keys.get('region')
            if merge_key and merge_key in merged.columns and merge_key in self.region_data.columns:
                merged = pd.merge(merged, self.region_data, on=merge_key, how='left', suffixes=('', '_region'))
                print(f"Merged with additional data: {len(merged)} records")
            else:
                print("Skipping additional data merge - no suitable key found") 
        self.merged_data = merged
        print(f"\nFinal merged dataset: {len(merged)} records with {len(merged.columns)} columns")
        # Display merged data sample 
        print("\nSample of merged data:")
        print(merged.head())
        print("Data Merging Process is Successfully implemented")
        return merged
    # Insights generation based on Sales data ( Well , mostly this area doesn't disturb but if it does - You GOT a problem )
    def generate_sales_data_insights(self):
        """Generate insights based on available data"""
        if self.merged_data is None:
            print("No merged data available for analysis")
            return
        print("\n" + "="*50)
        print("AUTO-GENERATING INSIGHTS")
        print("="*50)
        sales_Metadata = self.merged_data
        insights = {}
        # Basic statistics
        print("Analyzing data structure...")
        insights['basic_stats'] = {
            'total_records': len(sales_Metadata),
            'total_columns': len(sales_Metadata.columns),
            'numeric_columns': len(sales_Metadata.select_dtypes(include=[np.number]).columns),
            'text_columns': len(sales_Metadata.select_dtypes(include=['object']).columns),
            'date_columns': len(sales_Metadata.select_dtypes(include=['datetime']).columns)
        }
        # Find numeric columns for analysis ( this region is like a colour festival in vscode )
        numeric_cols = sales_Metadata.select_dtypes(include=[np.number]).columns.tolist()
        text_cols = sales_Metadata.select_dtypes(include=['object']).columns.tolist()
        date_cols = sales_Metadata.select_dtypes(include=['datetime']).columns.tolist()
        print(f"Found {len(numeric_cols)} numeric columns: {numeric_cols}")
        print(f"Found {len(text_cols)} text columns: {text_cols}")
        print(f"Found {len(date_cols)} date columns: {date_cols}")
        # Revenue/Sales analysis ( look for money-related columns )
        revenue_cols = [col for col in numeric_cols if any(keyword in col.lower() for keyword in ['revenue', 'sales', 'amount', 'price', 'cost', 'total'])]
        if revenue_cols:
            insights['financial_summary'] = {}
            for col in revenue_cols:
                insights['financial_summary'][col] = {
                    'total': sales_Metadata[col].sum(),
                    'average': sales_Metadata[col].mean(),
                    'median': sales_Metadata[col].median(),
                    'min': sales_Metadata[col].min(),
                    'max': sales_Metadata[col].max()
                }
        # Categorical analysis
        categorical_insights = {}
        for col in text_cols:
            if sales_Metadata[col].nunique() < 20:  # Only analyze columns with reasonable number of categories
                value_counts = sales_Metadata[col].value_counts()
                categorical_insights[col] = {
                    'unique_values': sales_Metadata[col].nunique(),
                    'top_values': value_counts.head().to_dict(),
                    'distribution': value_counts.to_dict()
                }
        if categorical_insights:
            insights['categorical_analysis'] = categorical_insights
        # Time series analysis (if date columns exist )
        if date_cols and revenue_cols:
            insights['time_analysis'] = {}
            for date_col in date_cols[:1]:  # Analyze first date column
                for rev_col in revenue_cols[:1]:  # Analyze first revenue column
                    df_time = sales_Metadata.groupby(pd.Grouper(key=date_col, freq='D'))[rev_col].sum().reset_index()
                    insights['time_analysis'][f'{rev_col}_by_{date_col}'] = {
                        'daily_average': df_time[rev_col].mean(),
                        'best_day': df_time.loc[df_time[rev_col].idxmax(), date_col] if len(df_time) > 0 else None,
                        'worst_day': df_time.loc[df_time[rev_col].idxmin(), date_col] if len(df_time) > 0 else None
                    }
        # Correlation analysis
        if len(numeric_cols) > 1:
            correlation_matrix = sales_Metadata[numeric_cols].corr()
            # Find strong correlations (> 0.7 or < -0.7)
            strong_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:
                        strong_correlations.append({
                            'col1': correlation_matrix.columns[i],
                            'col2': correlation_matrix.columns[j],
                            'correlation': corr_value
                        })
            if strong_correlations:
                insights['correlations'] = strong_correlations
        self.insights = insights
        print("Insights generated successfully!")
    # Make Report of the Observed Insights
    def print_insights_report(self):
        """Print comprehensive insights report"""
        if not self.insights:
            print("No insights available. Call and Run auto_generate_insights() first...!")
            return
        print("\n" + "="*60)
        print("AUTOMATED BUSINESS INSIGHTS REPORT")
        print("="*60)
        # Basic Statistics Section
        if 'basic_stats' in self.insights:
            print("\nDATA OVERVIEW")
            print("-" * 20)
            stats = self.insights['basic_stats']
            for key, value in stats.items():
                print(f"{key.replace('_', ' ').title()}: {value}")
        # Financial Summary Section
        if 'financial_summary' in self.insights:
            print("\nFINANCIAL SUMMARY")
            print("-" * 25)
            for col, stats in self.insights['financial_summary'].items():
                print(f"\n{col.upper()}:")
                for metric, value in stats.items():
                    print(f"  {metric.title()}: {value:,.2f}")
        # Categorical Analysis Section
        if 'categorical_analysis' in self.insights:
            print("\nCATEGORICAL ANALYSIS")
            print("-" * 30)
            for col, analysis in self.insights['categorical_analysis'].items():
                print(f"\n{col.upper()}:")
                print(f"  Unique values: {analysis['unique_values']}")
                print("  Top categories:")
                for category, count in list(analysis['top_values'].items())[:5]:
                    print(f"    {category}: {count}")
        # Time Analysis Section ( this section could really go wrong )
        if 'time_analysis' in self.insights:
            print("\nTIME SERIES ANALYSIS")
            print("-" * 30)
            for analysis_name, data in self.insights['time_analysis'].items():
                print(f"\n{analysis_name.upper()}:")
                for metric, value in data.items():
                    print(f"  {metric.replace('_', ' ').title()}: {value}")
        # Correlations Section
        if 'correlations' in self.insights:
            print("\nSTRONG CORRELATIONS")
            print("-" * 25)
            for corr in self.insights['correlations']:
                print(f"{corr['col1']} ↔ {corr['col2']}: {corr['correlation']:.3f}")
    
    def create_sales_data_visualizations(self): # This function was challenging during implementation but now works as intended
        """Create visualizations based on available data"""
        if self.merged_data is None:
            print("No data available for visualization")
            return
        print("\n" + "="*50)
        print("GENERATING SALES DATA VISUALIZATIONS")
        print("="*50)
        sales_Metadata = self.merged_data
        numeric_cols = sales_Metadata.select_dtypes(include=[np.number]).columns.tolist()
        text_cols = sales_Metadata.select_dtypes(include=['object']).columns.tolist()
        date_cols = sales_Metadata.select_dtypes(include=['datetime']).columns.tolist()
        # Calculate number of subplots needed 
        n_plots = min(6, len(numeric_cols) + len(text_cols))
        if n_plots == 0:
            print("No suitable columns found for visualization")
            return
        # Set up the plotting environment
        fig = plt.figure(figsize=(15, 10))
        plot_count = 0
        # Plot numeric columns (histograms)
        for i, col in enumerate(numeric_cols[:3]):
            if plot_count >= 6:
                break
            plot_count += 1
            plt.subplot(2, 3, plot_count)
            plt.hist(sales_Metadata[col].dropna(), bins=20, alpha=0.7, color='skyblue')
            plt.title(f'Distribution of {col}', fontsize=12, fontweight='bold')
            plt.xlabel(col)
            plt.ylabel('Frequency')
        # Plot categorical columns (bar charts)
        for i, col in enumerate(text_cols[:3]):
            if plot_count >= 6:
                break
            if sales_Metadata[col].nunique() <= 10:  # Only plot if reasonable number of categories
                plot_count += 1
                plt.subplot(2, 3, plot_count)
                value_counts = sales_Metadata[col].value_counts().head(10)
                plt.bar(range(len(value_counts)), value_counts.values, color='lightcoral')
                plt.title(f'Top Values in {col}', fontsize=12, fontweight='bold')
                plt.ylabel('Count')
                plt.xticks(range(len(value_counts)), value_counts.index, rotation=45, ha='right')
        # Time series plot if available to do
        if date_cols and numeric_cols and plot_count < 6:
            plot_count += 1
            plt.subplot(2, 3, plot_count)
            date_col = date_cols[0]
            num_col = numeric_cols[0]
            # Group visualization data by date and sum
            time_data = sales_Metadata.groupby(pd.Grouper(key=date_col, freq='D'))[num_col].sum()
            plt.plot(time_data.index, time_data.values, marker='o', linewidth=2)
            plt.title(f'{num_col} Over Time', fontsize=12, fontweight='bold')
            plt.xlabel('Date')
            plt.ylabel(num_col)
            plt.xticks(rotation=45)
        # Correlation heatmap if multiple numeric columns
        if len(numeric_cols) > 1 and plot_count < 6:
            plot_count += 1
            plt.subplot(2, 3, plot_count)
            correlation_matrix = sales_Metadata[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
            plt.title('Correlation Matrix', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig('auto_insights_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show() # Make sure this plt.show() is working , This here could Ragebait you to your limits 
        print("Visualizations created and saved as 'auto_insights_dashboard.png'")
    
    def run_custom_pipeline(self): # This def function really played  with me
        """Run the complete data pipeline with your files"""
        print("STARTING CUSTOM DATA PIPELINE")
        print("=" * 60)
        # Inspect your files first
        self.inspect_data_files()
        # Load data from your files
        print("\nLOADING YOUR DATA FILES")
        print("-" * 35)
        if self.csv_file:
            self.users_data = self.load_csv_file_data()
        if self.json_file:
            self.metadata = self.load_json_file_data()
        if self.excel_file:
            self.region_data = self.load_excel_file_data()
        # Check if at least one file was loaded
        if all(data is None for data in [self.users_data, self.metadata, self.region_data]):
            print("No data was successfully loaded. Please check your file paths and formats.")
            return
        # Use the largest dataset as primary data
        datasets = []
        if self.users_data is not None:
            datasets.append(('CSV', self.users_data))
        if self.metadata is not None:
            datasets.append(('JSON', self.metadata))
        if self.region_data is not None:
            datasets.append(('Excel', self.region_data))
        # Sort by size and use largest one there is as primary
        datasets.sort(key=lambda x: len(x[1]), reverse=True)
        primary_type, self.users_data = datasets[0]
        print(f"\nUsing {primary_type} data as primary dataset ({len(self.users_data)} records)")
        # Reassign other available datasets
        if len(datasets) > 1:
            if primary_type != 'JSON' and any(t[0] == 'JSON' for t in datasets[1:]):
                self.metadata = next(t[1] for t in datasets[1:] if t[0] == 'JSON')
            if primary_type != 'Excel' and any(t[0] == 'Excel' for t in datasets[1:]):
                self.region_data = next(t[1] for t in datasets[1:] if t[0] == 'Excel')
        # Merge data sources
        self.sales_merge_data()
        # Generate insights on the data which is ready
        self.generate_sales_data_insights()
        # Print Sales report
        self.print_insights_report()
        # Create Sales data visualizations ( Handled by User)
        response = input("Want to see the visualizations? (yes/no): ").strip().lower()
        if response == 'yes':
            print(" Visualization Sequence Initiated..... PLease wait till the plots are loaded ")
            time.sleep(2) # That one time you see this time import in use 
            print(" Visualizations are generated ")
            self.create_sales_data_visualizations()
        else:
            print("Skipping Visualization !!! Returning to Terminal")
        # Export the Finished results 
        self.export_custom_results()
        print("\n" + "="*60)
        print("CUSTOM PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nFiles generated:")
        print("- auto_insights_dashboard.png (Visualizations)")
        print("- custom_merged_data.csv (Processed data)")
        print("- custom_insights_report.txt (Text insights)")
    
    def export_custom_results(self): # The only Def function that didn't slap my face with errors
        """Export results from your custom data"""
        print("\nEXPORTING CUSTOM RESULTS")
        print("-" * 30)
        # Export merged data if available
        if self.merged_data is not None:
            self.merged_data.to_csv('custom_merged_data.csv', index=False)
            print("Merged data exported to 'custom_merged_data.csv'")
        # Export insights to text file
        if self.insights:
            with open('custom_insights_report.txt', 'w') as f:
                f.write("CUSTOM DATA INSIGHTS REPORT\n")
                f.write("=" * 50 + "\n\n")
                for section, data in self.insights.items():
                    f.write(f"{section.upper().replace('_', ' ')}\n")
                    f.write("-" * 30 + "\n")
                    f.write(str(data))
                    f.write("\n\n")
            print("Insights exported to 'custom_insights_report.txt'")

# Example usage with your own Sample or Heavy-Data files
if __name__ == "__main__":
    print("CUSTOM DATA PIPELINE SETUP")
    print("=" * 40)
    print(f"Note : Pipeline execution started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # MODIFY THESE PATHS TO YOUR ACTUAL FILES 
    your_csv_file = r"c:\Users\yaswa\OneDrive\Desktop\Flipkart Project\multisource_sales_dashboard_demo\users_data.csv"      # Replace with your CSV file path
    your_json_file = r"c:\Users\yaswa\OneDrive\Desktop\Flipkart Project\multisource_sales_dashboard_demo\mcc_code.json"      # Replace with your JSON file path  
    your_excel_file = r"c:\Users\yaswa\OneDrive\Desktop\Flipkart Project\multisource_sales_dashboard_demo\region_info.xlsx"      # Replace with your Excel file path
    # Initialize the pipeline with your own files
    pipeline = RobustSalesDataPipeline(
        csv_file=your_csv_file,
        json_file=your_json_file, 
        excel_file=your_excel_file
    )
    # Now Run the completed pipeline - Happiness can't be explained
    pipeline.run_custom_pipeline()
    
    print("\n Data Pipeline is successfully Executed and Task-1 Completed !")
