import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import shap

def run_optimized_pipeline(csv_path):
    # 1. Load Data (Using a flexible path)
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found in this folder.")
        return

    # 2. Standardize & Clean
    df.columns = [c.lower().replace(' ', '_').replace('-', '_').replace('%', '') for c in df.columns]
    
    # 3. Feature Engineering
    # 'mixed' format handles AM/PM issues automatically
    df['hour'] = pd.to_datetime(df['time'], format='mixed').dt.hour

    # 4. Features Selection (Avoiding Leakage)
    features = ['branch', 'city', 'customer_type', 'gender', 'product_line', 'unit_price', 'quantity', 'hour']
    X = df[features].copy()
    y = df['sales']

    # 5. Mappings & Encoding
    mappings = {
        "branch": {"Alex": 0, "Cairo": 1, "Giza": 2},
        "city": {"Mandalay": 0, "Naypyitaw": 1, "Yangon": 2},
        "customer_type": {"Member": 0, "Normal": 1},
        "gender": {"Female": 0, "Male": 1},
        "product_line": {
            "Electronic accessories": 0, "Fashion accessories": 1, 
            "Food and beverages": 2, "Health and beauty": 3, 
            "Home and lifestyle": 4, "Sports and travel": 5
        }
    }

    for col in mappings.keys():
        X[col] = X[col].map(mappings[col])

    # 6. Train-Test Split & Modeling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 7. SHAP Global Explainability
    print("Generating SHAP plots...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=features, show=False)
    plt.savefig('shap_summary.png', bbox_inches='tight')
    plt.close()

    # 8. Save Final Assets
    joblib.dump(model, 'supermarket_sales_model.pkl')
    joblib.dump(mappings, 'category_mappings.pkl')
    
    # Final Metrics
    y_pred = model.predict(X_test)
    print(f"Final R2 Score: {r2_score(y_test, y_pred):.4f}")
    print(f"Mean Absolute Error: ${mean_absolute_error(y_test, y_pred):.2f}")
    print("--- Pipeline Complete: All files saved! ---")

if __name__ == "__main__":
    run_optimized_pipeline('SuperMarket Analysis (1).csv')