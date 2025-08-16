📈 Stock Price Prediction Dashboard

An interactive Streamlit dashboard to predict future stock prices using Facebook Prophet and stock data from Yahoo Finance.

📌 Table of Contents

Features

Tech Stack

Installation

Usage

Project Structure

Screenshots

Requirements

Acknowledgements

🚀 Features

✅ Fetch historical stock data from Yahoo Finance

✅ Forecast future prices using Prophet

✅ Visualize actual vs. predicted stock prices

✅ Display confidence intervals for predictions

✅ Show model performance metrics:

MAPE (Mean Absolute Percentage Error)

RMSE (Root Mean Squared Error)

MSE (Mean Squared Error)

✅ Trend & seasonality decomposition

✅ Export forecast results as CSV

✅ Simple, interactive Streamlit UI

🛠️ Tech Stack

Python

Streamlit – Web UI

yfinance – Stock data

Prophet – Forecasting model

Matplotlib – Visualization

scikit-learn – Metrics

Pandas & NumPy – Data handling

⚙️ Installation

Clone the repository

git clone https://github.com/your-username/stock-price-prediction-dashboard.git
cd stock-price-prediction-dashboard


Create a virtual environment (recommended)

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows


Install dependencies

pip install -r requirements.txt

📊 Usage

Run the app:

streamlit run app.py


Enter a stock ticker (e.g., AAPL, MSFT, GOOGL)

Select a date range

Choose days to predict

Click Run Prediction

Download forecast results as CSV

📂 Project Structure
📁 stock-price-prediction-dashboard
│── app.py                # Main Streamlit application
│── requirements.txt       # Dependencies
│── README.md              # Documentation
