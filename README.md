#  Robust Multi-Source Sales Data Pipeline

>  Built by **Vajja Sri Nikitha** for Flipkart Task-1  
> 🗓 Last Updated: **30 July 2025**

---

##  Overview

## 📊 Custom Sales Data Pipeline & Insights Dashboard

This project implements a **custom sales data pipeline** that ingests, cleans, analyzes, and visualizes sales-related data from multiple sources including **CSV, JSON, and Excel**.

### 🎯 Designed To Be:

* **Modular** – Easy to extend or adapt
* **Insight-rich** – Automatically extracts useful KPIs
* **Visualization-capable** – Generates visual dashboards
* **Production-ready** – Robust structure and exportable results

---

## 📁 File Structure

```plaintext
Sales_Data_Pipeline_Main.py      # Main pipeline script
custom_merged_data.csv           # Auto-generated merged dataset
auto_insights_dashboard.png      # Auto-generated dashboard visuals
custom_insights_report.txt       # Auto-generated text report
```

---

## ✅ Features

* ✅ Multi-source data ingestion (CSV, JSON, Excel)
* ✅ Auto-cleaning & intelligent type conversion
* ✅ Smart merging using detected keys
* ✅ Insight extraction (summary stats, time series, category analysis)
* ✅ Visualizations (bar charts, scatter plots, pie charts, heatmaps)
* ✅ Export outputs (CSV, PNG, TXT)
* ✅ Handles real-world dirty data (missing columns, type issues)

---

## 🔧 Requirements

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn tabulate openpyxl
```

---

## 🚀 Usage

### 1. Configure File Paths

Edit these lines in the `__main__` block:

```python
your_csv_file = r"path\to\sales_data.csv"
your_json_file = r"path\to\product_metadata.json"
your_excel_file = r"path\to\region_info.xlsx"
```

### 2. Run the Pipeline

```bash
python Sales_Data_Pipeline_Main.py
```

When prompted, choose whether to generate visualizations.

---

## 📦 Output Files

After successful execution:

* `custom_merged_data.csv` – Cleaned and merged dataset
* `custom_insights_report.txt` – Human-readable analysis report
* `auto_insights_dashboard.png` – Visual dashboard image

---

## 📈 Example Insights Extracted

**DATA OVERVIEW**

* Total Records: 2000
* Columns: 14 (Numeric: 9, Text: 5, Date: 0)

**CATEGORY ANALYSIS**

* GENDER Breakdown → Female: 1016 | Male: 984

**Metrics Extracted:**

* Total revenue, average cost, margin %
* Top-selling products and regions
* Sales trends by date
* Correlations between metrics

---

## 🖼️ Sample Visuals

### Flowchart Overview

![Flowchart](images/flowchart.png)

### Sales Insights Dashboard

![Dashboard](images/dashboard.png)

### Analytics & Visualizations

![Sample Visuals](images/sample_visuals.png)

> Replace the file paths with your actual paths if hosted on GitHub or locally.

---

## 🛠️ Error Handling

Robust error catching for:

* Missing/corrupt files
* Unsupported formats
* Merge issues
* Type conversion warnings

---

## 🔄 Version Control

This project is tracked via Git.
✅ All core scripts and artifacts are versioned
✅ Production-ready commits are labeled
✅ Clean, AI-refactored codebase

---

##  Author

**Vajja Sri Nikitha**  
*Flipkart Data Pipeline Task-1 Project*

