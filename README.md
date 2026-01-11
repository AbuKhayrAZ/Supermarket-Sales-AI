# 🛒 Supermarket Sales Prediction AI

## Project Overview
This project uses Machine Learning (Random Forest) to predict supermarket sales with **99.8% $R^2$ accuracy**. It includes an interactive dashboard for "What-If" business analysis.

## 🚀 How to Run
1. **Train the Model:** Run `python main.py` to process the data and generate the model files.
2. **Launch the App:** Run `streamlit run app.py` to open the interactive dashboard in your browser.

## 📊 Key Features
- **Predictive Engine:** Forecasts total sales based on Branch, Product Line, and Quantity.
- **Explainable AI (SHAP):** Visualizes exactly why the model made a specific prediction.
- **Business Insights:** Automatically suggests strategies (e.g., bundle discounts) based on predicted value.

## 🛠️ Requirements
- Python 3.8+
- Libraries: `pandas`, `scikit-learn`, `streamlit`, `joblib`, `shap`, `matplotlib`