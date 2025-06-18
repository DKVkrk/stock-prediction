import yfinance as yf
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Stock Price Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .st-bq {
        border-left: 4px solid #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("📈 Stock Price Prediction Dashboard")
st.markdown("Predict future stock prices using Facebook Prophet and analyze model performance.")

# ================== Helper Functions ==================
def fetch_stock_data(ticker, start_date, end_date):
    """Fetch stock data from Yahoo Finance."""
    df = yf.download(ticker, start=start_date, end=end_date)
    df.reset_index(inplace=True)
    df = df[['Date', 'Close']]
    df.columns = ['ds', 'y']
    df['ds'] = df['ds'].dt.tz_localize(None)
    df['moving_avg'] = df['y'].rolling(window=5).mean().fillna(method='bfill')
    return df

def predict_stock_price(ticker, days, start_date, end_date):
    """Train Prophet model and make predictions."""
    df = fetch_stock_data(ticker, start_date, end_date)
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
    model.add_regressor('moving_avg')
    model.fit(df)
    
    future = model.make_future_dataframe(periods=days)
    future['moving_avg'] = df['y'].iloc[-5:].mean()
    forecast = model.predict(future)
    
    return df, forecast, model

def calculate_metrics(df, forecast):
    """Calculate MAPE, MSE, RMSE."""
    actual = df['y'].values
    predicted = forecast['yhat'].values[:len(actual)]
    
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    
    return mape, mse, rmse

# ================== Sidebar (User Inputs) ==================
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Ticker input
    ticker = st.text_input("Stock Ticker (e.g., AAPL, MSFT, GOOGL)", "AAPL").upper()
    
    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime(2015, 1, 1))
    with col2:
        end_date = st.date_input("End Date", datetime(2022, 1, 1))
    
    # Days to predict
    days_to_predict = st.slider("Days to Predict", 1, 365, 30)
    
    # Additional options
    st.markdown("### 📊 Display Options")
    show_components = st.checkbox("Show Trend/Seasonality", True)
    show_raw_data = st.checkbox("Show Raw Data", False)

# ================== Main Dashboard ==================
if st.sidebar.button("Run Prediction"):
    try:
        with st.spinner("Training model and generating predictions..."):
            df, forecast, model = predict_stock_price(ticker, days_to_predict, start_date, end_date)
            mape, mse, rmse = calculate_metrics(df, forecast)
        
        # ========== 1. Prediction Plot ==========
        st.subheader(f"📉 {ticker} Stock Price Forecast")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['ds'], df['y'], label='Actual Price', color='blue')
        ax.plot(forecast['ds'], forecast['yhat'], label='Predicted Price', color='orange')
        ax.fill_between(
            forecast['ds'],
            forecast['yhat_lower'],
            forecast['yhat_upper'],
            color='orange',
            alpha=0.2,
            label='Confidence Interval'
        )
        ax.axvline(df['ds'].iloc[-1], color='red', linestyle='--', label='Training End')
        ax.set_xlabel("Date")
        ax.set_ylabel("Price ($)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        
        # ========== 2. Key Metrics ==========
        st.subheader("📊 Model Performance")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Absolute % Error (MAPE)", f"{mape:.2f}%")
        with col2:
            st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")
        with col3:
            st.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
        
        # ========== 3. Trend & Seasonality ==========
        if show_components:
            st.subheader("📅 Trend & Seasonality Analysis")
            fig_components = model.plot_components(forecast)
            st.pyplot(fig_components)
        
        # ========== 4. Raw Data ==========
        if show_raw_data:
            st.subheader("📂 Raw Data")
            
            tab1, tab2 = st.tabs(["Historical Data", "Forecast Data"])
            with tab1:
                st.dataframe(df)
            with tab2:
                st.dataframe(forecast)
        
        # ========== 5. Download Predictions ==========
        st.subheader("💾 Export Data")
        csv = forecast.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Forecast (CSV)",
            data=csv,
            file_name=f"{ticker}_forecast.csv",
            mime="text/csv"
        )
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
else:
    st.info("👈 Enter a stock ticker and click **Run Prediction** to start.")

# Footer
st.markdown("---")
st.markdown("🔍 *Built with Streamlit, yfinance, and Prophet*")






