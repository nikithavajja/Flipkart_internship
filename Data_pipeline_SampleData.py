#!/usr/bin/env python3
"""
Multi-Source Sales Data Pipeline & Business Insights Dashboard
Student Project - Data Engineering & Analytics

This project demonstrates:
1. Multi-format data ingestion (CSV, JSON, Excel)
2. Data cleaning and transformation
3. Data merging and integration
4. Business insights generation
5. Visualization and reporting

Required Libraries: pandas, numpy, matplotlib, seaborn, tabulate, openpyxl, json
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from datetime import datetime, timedelta
import warnings
import os
from pathlib import Path

# Configure display settings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SalesDataPipeline:
    """
    A comprehensive sales data pipeline that handles multiple data sources,
    performs data cleaning, transformation, and generates business insights.
    """
    
    def __init__(self):
        self.sales_data = None
        self.metadata = None
        self.region_data = None
        self.merged_data = None
        self.insights = {}
        
    def create_sample_data(self):
        """Create sample data files for demonstration"""
        print("Creating sample data files...")
        
        # Sample sales data (CSV)
        sales_data = {
            'Date': ['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04', '2025-01-05',
                    '2025-01-06', '2025-01-07', '2025-01-08', '2025-01-09', '2025-01-10'],
            'Product': ['Widget A', 'Widget B', 'Widget A', 'Widget C', 'Widget B',
                       'Widget A', 'Widget C', 'Widget B', 'Widget A', 'Widget C'],
            'Units Sold': [10, 5, 8, 12, 7, 15, 9, 6, 11, 14],
            'Revenue': [100.0, 75.0, 80.0, 144.0, 105.0, 150.0, 108.0, 90.0, 110.0, 168.0],
            'Region': ['North', 'South', 'North', 'East', 'South', 'West', 'East', 'South', 'North', 'West']
        }
        sales_df = pd.DataFrame(sales_data)
        sales_df.to_csv('sales_data.csv', index=False)
        
        # Sample product metadata (JSON)
        metadata = [
            {"Product": "Widget A", "Category": "Electronics", "Cost": 8.0, "Margin": 0.2},
            {"Product": "Widget B", "Category": "Accessories", "Cost": 12.0, "Margin": 0.25},
            {"Product": "Widget C", "Category": "Electronics", "Cost": 10.0, "Margin": 0.2}
        ]
        with open('product_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Sample region info (Excel)
        region_data = {
            'Region': ['North', 'South', 'East', 'West'],
            'Manager': ['John Smith', 'Sarah Johnson', 'Mike Wilson', 'Lisa Brown'],
            'Target': [1000, 1200, 800, 900],
            'Population': [500000, 750000, 400000, 600000]
        }
        region_df = pd.DataFrame(region_data)
        region_df.to_excel('region_info.xlsx', index=False)
        
        print("✓ Sample data files created successfully!")
    
    def load_csv_data(self, filepath):
        """Load and validate CSV data"""
        try:
            print(f"Loading CSV data from {filepath}...")
            data = pd.read_csv(filepath)
            
            # Data validation
            required_columns = ['Date', 'Product', 'Units Sold', 'Revenue', 'Region']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Clean and convert data types
            data['Date'] = pd.to_datetime(data['Date'])
            data['Units Sold'] = pd.to_numeric(data['Units Sold'], errors='coerce')
            data['Revenue'] = pd.to_numeric(data['Revenue'], errors='coerce')
            
            # Remove any rows with null values in critical columns
            data = data.dropna(subset=['Units Sold', 'Revenue'])
            
            print(f"✓ Loaded {len(data)} records from CSV")
            return data
            
        except Exception as e:
            print(f"✗ Error loading CSV: {str(e)}")
            return None
    
    def load_json_data(self, filepath):
        """Load and validate JSON metadata"""
        try:
            print(f"Loading JSON data from {filepath}...")
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Validate required columns
            required_columns = ['Product', 'Category']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            print(f"✓ Loaded {len(df)} product metadata records from JSON")
            return df
            
        except Exception as e:
            print(f"✗ Error loading JSON: {str(e)}")
            return None
    
    def load_excel_data(self, filepath):
        """Load and validate Excel data"""
        try:
            print(f"Loading Excel data from {filepath}...")
            data = pd.read_excel(filepath)
            
            # Validate required columns
            required_columns = ['Region']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            print(f"✓ Loaded {len(data)} region records from Excel")
            return data
            
        except Exception as e:
            print(f"✗ Error loading Excel: {str(e)}")
            return None
    
    def merge_data_sources(self):
        """Merge all data sources into a unified dataset"""
        print("\n" + "="*50)
        print("MERGING DATA SOURCES")
        print("="*50)
        
        if self.sales_data is None:
            print("✗ No sales data available for merging")
            return None
        
        merged = self.sales_data.copy()
        
        # Merge with metadata (left join to keep all sales records)
        if self.metadata is not None:
            print("Merging sales data with product metadata...")
            merged = pd.merge(merged, self.metadata, on='Product', how='left')
            print(f"✓ Merged with metadata: {len(merged)} records")
        
        # Merge with region data
        if self.region_data is not None:
            print("Merging with region information...")
            merged = pd.merge(merged, self.region_data, on='Region', how='left')
            print(f"✓ Merged with region data: {len(merged)} records")
        
        # Calculate additional metrics
        merged['Unit Price'] = merged['Revenue'] / merged['Units Sold']
        
        if 'Cost' in merged.columns:
            merged['Profit'] = merged['Revenue'] - (merged['Cost'] * merged['Units Sold'])
            merged['Profit Margin %'] = (merged['Profit'] / merged['Revenue'] * 100).round(2)
        
        self.merged_data = merged
        print(f"✓ Final merged dataset: {len(merged)} records with {len(merged.columns)} columns")
        return merged
    
    def generate_insights(self):
        """Generate comprehensive business insights"""
        if self.merged_data is None:
            print("No merged data available for analysis")
            return
        
        print("\n" + "="*50)
        print("GENERATING BUSINESS INSIGHTS")
        print("="*50)
        
        df = self.merged_data
        
        # 1. Summary Statistics
        summary_stats = {
            'total_revenue': df['Revenue'].sum(),
            'total_units': df['Units Sold'].sum(),
            'total_orders': len(df),
            'avg_order_value': df['Revenue'].mean(),
            'date_range': f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}"
        }
        
        # 2. Product Performance
        product_performance = df.groupby('Product').agg({
            'Revenue': 'sum',
            'Units Sold': 'sum',
            'Date': 'count'
        }).rename(columns={'Date': 'Order Count'}).round(2)
        product_performance['Avg Revenue per Order'] = (product_performance['Revenue'] / product_performance['Order Count']).round(2)
        
        # 3. Regional Performance
        regional_performance = df.groupby('Region').agg({
            'Revenue': 'sum',
            'Units Sold': 'sum',
            'Date': 'count'
        }).rename(columns={'Date': 'Order Count'}).round(2)
        
        if 'Target' in df.columns:
            regional_targets = df.groupby('Region')['Target'].first()
            regional_performance['Target'] = regional_targets
            regional_performance['Target Achievement %'] = (regional_performance['Revenue'] / regional_performance['Target'] * 100).round(2)
        
        # 4. Category Analysis
        if 'Category' in df.columns:
            category_performance = df.groupby('Category').agg({
                'Revenue': 'sum',
                'Units Sold': 'sum',
                'Date': 'count'
            }).rename(columns={'Date': 'Order Count'}).round(2)
        else:
            category_performance = None
        
        # 5. Time Series Analysis
        daily_sales = df.groupby('Date').agg({
            'Revenue': 'sum',
            'Units Sold': 'sum'
        }).round(2)
        
        # 6. Profitability Analysis (if cost data available)
        if 'Profit' in df.columns:
            profitability = df.groupby('Product').agg({
                'Profit': 'sum',
                'Profit Margin %': 'mean'
            }).round(2)
        else:
            profitability = None
        
        # Store insights
        self.insights = {
            'summary': summary_stats,
            'products': product_performance,
            'regions': regional_performance,
            'categories': category_performance,
            'daily_trends': daily_sales,
            'profitability': profitability
        }
        
        print("✓ Business insights generated successfully!")
    
    def print_insights_report(self):
        """Print comprehensive insights report"""
        if not self.insights:
            print("No insights available. Run generate_insights() first.")
            return
        
        print("\n" + "="*60)
        print("BUSINESS INSIGHTS REPORT")
        print("="*60)
        
        # Summary Statistics
        print("\n📊 EXECUTIVE SUMMARY")
        print("-" * 30)
        summary = self.insights['summary']
        print(f"📈 Total Revenue: ${summary['total_revenue']:,.2f}")
        print(f"📦 Total Units Sold: {summary['total_units']:,}")
        print(f"🛒 Total Orders: {summary['total_orders']:,}")
        print(f"💰 Average Order Value: ${summary['avg_order_value']:.2f}")
        print(f"📅 Date Range: {summary['date_range']}")
        
        # Product Performance
        print("\n🏆 TOP PERFORMING PRODUCTS")
        print("-" * 35)
        products = self.insights['products'].sort_values('Revenue', ascending=False)
        print(tabulate(products, headers=products.columns, tablefmt='grid', floatfmt='.2f'))
        
        # Regional Performance
        print("\n🌍 REGIONAL PERFORMANCE")
        print("-" * 30)
        regions = self.insights['regions'].sort_values('Revenue', ascending=False)
        print(tabulate(regions, headers=regions.columns, tablefmt='grid', floatfmt='.2f'))
        
        # Category Performance
        if self.insights['categories'] is not None:
            print("\n📂 CATEGORY ANALYSIS")
            print("-" * 25)
            categories = self.insights['categories'].sort_values('Revenue', ascending=False)
            print(tabulate(categories, headers=categories.columns, tablefmt='grid', floatfmt='.2f'))
        
        # Profitability Analysis
        if self.insights['profitability'] is not None:
            print("\n💸 PROFITABILITY ANALYSIS")
            print("-" * 30)
            profit = self.insights['profitability'].sort_values('Profit', ascending=False)
            print(tabulate(profit, headers=profit.columns, tablefmt='grid', floatfmt='.2f'))
    
    def create_visualizations(self):
        """Create comprehensive data visualizations"""
        if self.merged_data is None or not self.insights:
            print("No data available for visualization")
            return
        
        print("\n" + "="*50)
        print("GENERATING VISUALIZATIONS")
        print("="*50)
        
        # Set up the plotting environment
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Revenue by Product (Bar Chart)
        plt.subplot(3, 3, 1)
        product_revenue = self.insights['products']['Revenue'].sort_values(ascending=True)
        plt.barh(product_revenue.index, product_revenue.values, color='skyblue')
        plt.title('Revenue by Product', fontsize=14, fontweight='bold')
        plt.xlabel('Revenue ($)')
        
        # 2. Units Sold by Region (Pie Chart)
        plt.subplot(3, 3, 2)
        region_units = self.insights['regions']['Units Sold']
        plt.pie(region_units.values, labels=region_units.index, autopct='%1.1f%%', startangle=90)
        plt.title('Units Sold by Region', fontsize=14, fontweight='bold')
        
        # 3. Daily Sales Trend (Line Chart)
        plt.subplot(3, 3, 3)
        daily_data = self.insights['daily_trends']
        plt.plot(daily_data.index, daily_data['Revenue'], marker='o', linewidth=2, markersize=6)
        plt.title('Daily Revenue Trend', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Revenue ($)')
        plt.xticks(rotation=45)
        
        # 4. Revenue vs Units Sold Scatter Plot
        plt.subplot(3, 3, 4)
        df = self.merged_data
        plt.scatter(df['Units Sold'], df['Revenue'], alpha=0.6, s=60)
        plt.title('Revenue vs Units Sold', fontsize=14, fontweight='bold')
        plt.xlabel('Units Sold')
        plt.ylabel('Revenue ($)')
        
        # 5. Category Revenue Distribution (if available)
        if self.insights['categories'] is not None:
            plt.subplot(3, 3, 5)
            cat_revenue = self.insights['categories']['Revenue']
            plt.bar(cat_revenue.index, cat_revenue.values, color='lightcoral')
            plt.title('Revenue by Category', fontsize=14, fontweight='bold')
            plt.ylabel('Revenue ($)')
            plt.xticks(rotation=45)
        
        # 6. Regional Target Achievement (if available)
        if 'Target Achievement %' in self.insights['regions'].columns:
            plt.subplot(3, 3, 6)
            achievement = self.insights['regions']['Target Achievement %']
            colors = ['green' if x >= 100 else 'orange' if x >= 80 else 'red' for x in achievement.values]
            plt.bar(achievement.index, achievement.values, color=colors)
            plt.title('Target Achievement by Region (%)', fontsize=14, fontweight='bold')
            plt.ylabel('Achievement %')
            plt.axhline(y=100, color='black', linestyle='--', alpha=0.7)
            plt.xticks(rotation=45)
        
        # 7. Product Profitability (if available)
        if self.insights['profitability'] is not None:
            plt.subplot(3, 3, 7)
            profit_margin = self.insights['profitability']['Profit Margin %']
            plt.bar(profit_margin.index, profit_margin.values, color='gold')
            plt.title('Profit Margin by Product (%)', fontsize=14, fontweight='bold')
            plt.ylabel('Profit Margin (%)')
            plt.xticks(rotation=45)
        
        # 8. Revenue Heatmap by Product and Region
        plt.subplot(3, 3, 8)
        if len(df['Product'].unique()) > 1 and len(df['Region'].unique()) > 1:
            pivot_data = df.pivot_table(values='Revenue', index='Product', columns='Region', aggfunc='sum', fill_value=0)
            sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlOrRd')
            plt.title('Revenue Heatmap: Product vs Region', fontsize=14, fontweight='bold')
        
        # 9. Monthly Summary (if data spans multiple months)
        plt.subplot(3, 3, 9)
        df['Month'] = df['Date'].dt.to_period('M')
        monthly_data = df.groupby('Month')['Revenue'].sum()
        if len(monthly_data) > 1:
            plt.bar(range(len(monthly_data)), monthly_data.values, color='mediumpurple')
            plt.title('Monthly Revenue', fontsize=14, fontweight='bold')
            plt.xlabel('Month')
            plt.ylabel('Revenue ($)')
            plt.xticks(range(len(monthly_data)), [str(m) for m in monthly_data.index], rotation=45)
        else:
            plt.text(0.5, 0.5, 'Insufficient data\nfor monthly analysis', 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=plt.gca().transAxes, fontsize=12)
            plt.title('Monthly Analysis', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('sales_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Visualizations created and saved as 'sales_dashboard.png'")
    
    def export_results(self):
        """Export results to various formats"""
        print("\n" + "="*50)
        print("EXPORTING RESULTS")
        print("="*50)
        
        if self.merged_data is None:
            print("No data to export")
            return
        
        # Export merged data to CSV
        self.merged_data.to_csv('merged_sales_data.csv', index=False)
        print("✓ Merged data exported to 'merged_sales_data.csv'")
        
        # Export insights to Excel with multiple sheets
        if self.insights:
            with pd.ExcelWriter('sales_insights.xlsx', engine='openpyxl') as writer:
                # Summary sheet
                summary_df = pd.DataFrame([self.insights['summary']]).T
                summary_df.columns = ['Value']
                summary_df.to_excel(writer, sheet_name='Summary')
                
                # Product performance
                self.insights['products'].to_excel(writer, sheet_name='Product Performance')
                
                # Regional performance
                self.insights['regions'].to_excel(writer, sheet_name='Regional Performance')
                
                # Category performance (if available)
                if self.insights['categories'] is not None:
                    self.insights['categories'].to_excel(writer, sheet_name='Category Performance')
                
                # Daily trends
                self.insights['daily_trends'].to_excel(writer, sheet_name='Daily Trends')
                
                # Profitability (if available)
                if self.insights['profitability'] is not None:
                    self.insights['profitability'].to_excel(writer, sheet_name='Profitability')
            
            print("✓ Insights exported to 'sales_insights.xlsx'")
    
    def run_pipeline(self, create_sample=True):
        """Run the complete data pipeline"""
        print("🚀 STARTING SALES DATA PIPELINE")
        print("=" * 60)
        
        # Create sample data if requested
        if create_sample:
            self.create_sample_data()
        
        # Load data from multiple sources
        print("\n📥 LOADING DATA SOURCES")
        print("-" * 30)
        
        self.sales_data = self.load_csv_data('sales_data.csv')
        self.metadata = self.load_json_data('product_metadata.json')
        self.region_data = self.load_excel_data('region_info.xlsx')
        
        # Merge data sources
        self.merge_data_sources()
        
        # Generate insights
        self.generate_insights()
        
        # Print report
        self.print_insights_report()
        
        # Create visualizations
        self.create_visualizations()
        
        # Export results
        self.export_results()
        
        print("\n" + "="*60)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nFiles generated:")
        print("- sales_dashboard.png (Visualizations)")
        print("- merged_sales_data.csv (Processed data)")
        print("- sales_insights.xlsx (Business insights)")
        print("\n📚 Ready for submission to GitHub!")

# Main execution
if __name__ == "__main__":
    # Initialize and run the pipeline
    pipeline = SalesDataPipeline()
    pipeline.run_pipeline(create_sample=True)
    
    # Additional analysis examples
    print("\n" + "="*50)
    print("ADDITIONAL ANALYSIS EXAMPLES")
    print("="*50)
    
    if pipeline.merged_data is not None:
        df = pipeline.merged_data
        
        # Example 1: Top performing product by region
        print("\n🏆 Top Performing Product by Region:")
        top_products = df.loc[df.groupby('Region')['Revenue'].idxmax()][['Region', 'Product', 'Revenue']]
        print(tabulate(top_products, headers=top_products.columns, tablefmt='grid', showindex=False))
        
        # Example 2: Sales velocity analysis
        print("\n⚡ Sales Velocity Analysis:")
        df['Sales Velocity'] = df['Revenue'] / df['Units Sold']
        velocity_analysis = df.groupby('Product')['Sales Velocity'].agg(['mean', 'std']).round(2)
        print(tabulate(velocity_analysis, headers=['Product', 'Avg Price', 'Price Std Dev'], tablefmt='grid'))
        
        # Example 3: Correlation analysis
        print("\n🔗 Correlation Analysis:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 2:
            correlation_matrix = df[numeric_cols].corr().round(3)
            print("Key correlations found in the dataset:")
            print(tabulate(correlation_matrix, headers=correlation_matrix.columns, tablefmt='grid'))
    
    print("\n🎓 Project completed! Perfect for your student portfolio.")
    print("Don't forget to:")
    print("- Add this to your GitHub repository")
    print("- Include a README.md with project description")
    print("- Document your findings and insights")
    print("- Add requirements.txt with all dependencies")