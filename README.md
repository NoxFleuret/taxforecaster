# 🏛️ TaxForecaster

> **Tax Revenue Forecasting & Scenario Planning System**
>
> *Advanced Analytics, Multi-Model Machine Learning, and AI-Driven Insights.*

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-ff4b4b.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 💻 Demo

**TaxForecaster** demo is availabe here https://taxforecaster.streamlit.app/

## 📖 Overview

**TaxForecaster** is a predictive analytics platform designed to assist fiscal policy analysts and decision-makers in collecting, forecasting, and analyzing tax revenue data.

By combining traditional econometric models (SARIMA, Holt-Winters) with modern machine learning (XGBoost, LightGBM, CatBoost, Prophet), the system provides robust revenue projections under various economic scenarios.

## ✨ Key Features

### 🚀 Advanced Forecasting Engine
- **Multi-Model Support**: Auto-selection or Ensemble of **15+ algorithms** including:
  - **Statistical**: SARIMA, Holt-Winters, Theta
  - **Machine Learning**: Random Forest, XGBoost, LightGBM, CatBoost
  - **Deep Learning**: LSTM (Recurrent Neural Networks)
  - **Time-Series**: Facebook Prophet
- **Auto-Tuning**: Integrated **Optuna** hyperparameter optimization for maximum accuracy.
- **Explainability**: SHAP value integration to understand feature drivers.

### 🎛️ Scenario Lab "What-If" Analysis
- Simulate the impact of macroeconomic shocks on tax revenue.
- Adjust key indicators: **ICP (Oil Price), GDP Growth, Inflation, Exchange Rate (USD), SBN Yields, Coal/CPO Prices**.
- Compare "Baseline" vs. "Crisis" vs. "Boom" scenarios instantly.

### 🛡️ Data Quality Center
- **Automated Validation**: Scans uploaded data for missing values, outliers, and schema errors.
- **Anomaly Detection**: Flags revenue spikes or drops that deviate from historical patterns.
- **Snapshots**: Maintain version history of your macro data sets.

### 📊 Interactive Dashboard
- **Executive Summary**: Real-time KPIs, Top Contributors, and Tax Buoyancy analysis.
- **Dynamic Reporting**: Generate shareable **HTML Executive Reports** with AI-generated narrative insights.
- **User Guide**: Built-in interactive tutorial and documentation.

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- pip

### Quick Start (Windows)

1.  **Clone the repository**
    ```powershell
    git clone https://github.com/NoxFleuret/taxforecaster.git
    cd taxforecaster
    ```

2.  **Install dependencies**
    ```powershell
    pip install -r requirements.txt
    ```

3.  **Run the application**
    ```powershell
    streamlit run Home.py
    ```

4.  **Access the Dashboard**
    Open your browser to `http://localhost:8501`

### 🐳 Docker Deployment

1.  **Build the image**
    ```bash
    docker build -t taxforecaster .
    ```

2.  **Run the container**
    ```bash
    docker run -p 8501:8501 taxforecaster
    ```

## 📂 Project Structure

A comprehensive guide to the file, folder, and module organization:

```text
tax_forecaster/
├── .gitignore               # Git exclude configuration
├── Dockerfile               # Docker container configuration
├── LICENSE                  # MIT License file
├── README.md                # Project documentation
├── requirements.txt         # Python library dependencies
├── Home.py                  # [APP ENTRY] Main application landing page
├── forecaster.py            # [CORE] Central Forecasting Engine class
├── config.yaml              # [CONF] Global configuration settings
├── config_loader.py         # [UTIL] Configuration loader module
│
├── pages/                   # [UI] Streamlit Pages (Sidebar Menu)
│   ├── 1_Dashboard.py       # Main Analytics & Forecasting Dashboard
│   ├── 1_Data_Quality.py    # Data Health, Validation & Snapshots
│   ├── 2_Scenario_Lab.py    # Economic Simulation Engine
│   ├── 2_Model_Lab.py       # Model Explainability (SHAP) & Tuning
│   ├── 3_Executive_Summary.py # High-level C-Suite View
│   └── 4_User_Guide.py      # Interactive Onboarding & Documentation
│
├── core/                    # [CORE] Backend logic modules
│   ├── data_loader.py       # Data ingestion & merging logic
│   └── feature_engineering.py # Lag features & rolling window calc
│
├── intelligence/            # [AI] Advanced analytics modules
│   ├── anomaly_detector.py  # Isolation Forest & Z-Score outliers
│   └── recommendation_engine.py # Actionable insights generator
│
├── tests/                   # [QA] Unit tests
│   ├── test_data_loader.py
│   ├── test_feature_engineering.py
│   ├── test_anomaly_detector.py
│   └── test_recommendation_engine.py
│
├── models/                  # [ML] Persisted model files (generated at runtime)
├── snapshots/               # [DATA] Version control storage for datasets
├── logs/                    # [SYS] Application execution logs
│
├── fetch_macro.py           # [DATA] Yahoo Finance & WorldBank API Integration
├── fetch_news.py            # [DATA] RSS News Feed Fetcher
├── data_validator.py        # [DATA] Schema validation & anomaly detection
├── data_versioning.py       # [DATA] Dataset snapshot management
├── report_generator.py      # [RPT] HTML Report generator logic
├── narrative_engine.py      # [AI] Rule-based text generation for reports
├── onboarding.py            # [UI] Interactive Wizard component
├── style.py                 # [UI] Global CSS styling & custom metric cards
├── theme_manager.py         # [UI] Theme switching logic
├── loading_animations.py    # [UI] Lottie & CSS loading animations
├── export_manager.py        # [UTIL] Logic for exporting results (ZIP/CSV)
├── error_handler.py         # [UTIL] Centralized error handling wrapper
├── logger.py                # [UTIL] Structured logging configuration
├── model_info.py            # [UTIL] Metadata & descriptions for algorithms
├── model_utils.py           # [UTIL] Helper functions for ML models
├── scenario_utils.py        # [UTIL] Scenario management helpers
├── scenario_presets.txt     # [CONF] Preset economic scenario definitions
└── [Data Files]             # Input CSVs (tax_history.csv, macro_data.csv)
```

## 🤝 Contributing

Contributions are welcome! Please read `CONTRIBUTING.md` for details on our code of conduct, and the process for submitting pull requests.

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

---

*Developed by Fasya_Dev for fiscal policy modernization initiatives.*







