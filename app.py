import streamlit as st
import pandas as pd
import joblib

# 1. Load trained assets
try:
    model = joblib.load('supermarket_sales_model.pkl')
    mappings = joblib.load('category_mappings.pkl')
except Exception as e:
    st.error(f"Missing model files! Run main.py first.")
    st.stop()

st.set_page_config(page_title="Sales AI Predictor", layout="wide")
st.title("🛒 Supermarket Strategy Tool")

# 2. Input Form (Sidebar)
with st.sidebar:
    st.header("Transaction Details")
    branch = st.selectbox("Branch", list(mappings['branch'].keys()))
    city = st.selectbox("City", list(mappings['city'].keys()))
    cust = st.selectbox("Customer Type", list(mappings['customer_type'].keys()))
    gender = st.selectbox("Gender", list(mappings['gender'].keys()))
    prod = st.selectbox("Product Line", list(mappings['product_line'].keys()))
    price = st.number_input("Unit Price ($)", value=50.0)
    qty = st.slider("Quantity", 1, 10, 5)
    hour = st.slider("Hour (24h)", 10, 21, 13)

# 3. Prediction Logic (Main Area)
if st.button("Generate Sales Forecast"):
    input_data = pd.DataFrame([[
        mappings['branch'][branch], mappings['city'][city],
        mappings['customer_type'][cust], mappings['gender'][gender],
        mappings['product_line'][prod], price, qty, hour
    ]], columns=['branch', 'city', 'customer_type', 'gender', 'product_line', 'unit_price', 'quantity', 'hour'])
    
    prediction = model.predict(input_data)[0]
    
    # --- ADD THE PROFESSIONAL TOUCHES HERE ---
    
    st.metric("Predicted Total Sale", f"${prediction:.2f}")
    
    # Business Insight Box
    if prediction > 500:
        st.warning("🚀 **Strategy Tip:** This is a high-value transaction. Ensure premium service or loyalty rewards are offered.")
    else:
        st.info("💡 **Strategy Tip:** Consider a 'Bundle Discount' (e.g., Buy 3 Get 1) to increase the quantity for this customer.")

    # Show the SHAP logic from the Master Script
    st.write("---")
    st.header("📊 Model Intelligence (Why this price?)")
    st.image("shap_summary.png", caption="SHAP Summary: This chart shows which factors influenced the prediction.")

    # Download Report Button
    report_text = f"Sales Forecast Report\nBranch: {branch}\nProduct: {prod}\nQuantity: {qty}\nPredicted Total: ${prediction:.2f}"
    st.download_button("📩 Download Prediction Report", report_text, file_name="forecast_report.txt")