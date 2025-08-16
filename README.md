# 📈 Stock Price Prediction Dashboard  

An interactive **Streamlit dashboard** to predict future stock prices using **Facebook Prophet** and stock data from **Yahoo Finance**.  

---

## 📌 Table of Contents  
1. [Features](#-features)  
2. [Tech Stack](#-tech-stack)  
3. [Installation](#-installation)  
4. [Usage](#-usage)  
5. [Project Structure](#-project-structure)   
7. [Requirements](#-requirements)  
8. [Acknowledgements](#-acknowledgements)  

---

## 🚀 Features  
- ✅ Fetch historical stock data from **Yahoo Finance**  
- ✅ Forecast future prices using **Prophet**  
- ✅ Visualize **actual vs. predicted** stock prices  
- ✅ Display **confidence intervals** for predictions  
- ✅ Show **model performance metrics**:  
  - MAPE (Mean Absolute Percentage Error)  
  - RMSE (Root Mean Squared Error)  
  - MSE (Mean Squared Error)  
- ✅ Trend & seasonality decomposition  
- ✅ Export forecast results as **CSV**  
- ✅ Simple, interactive **Streamlit UI**  

---

## 🛠️ Tech Stack  
- **Python**  
- **Streamlit** – Web UI  
- **yfinance** – Stock data  
- **Prophet** – Forecasting model  
- **Matplotlib** – Visualization  
- **scikit-learn** – Metrics  
- **Pandas & NumPy** – Data handling  

---

## ⚙️ Installation  

1. **Clone the repository**  
```bash
git clone https://github.com/your-username/stock-price-prediction-dashboard.git
cd stock-price-prediction-dashboard
Create a virtual environment (recommended)

bash
Copy
Edit
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
📊 Usage
Run the app:

bash
Copy
Edit
streamlit run app.py
Enter a stock ticker (e.g., AAPL, MSFT, GOOGL)

Select a date range

Choose days to predict

Click Run Prediction

Download forecast results as CSV

📂 Project Structure
bash
Copy
Edit
📁 stock-price-prediction-dashboard
│── app.py                # Main Streamlit application
│── requirements.txt       # Dependencies
│── README.md              # Documentation
📷 Screenshots
Dashboard Overview
(Add screenshot here)

Forecast Plot
(Add screenshot here)

📦 Requirements
Add the following to requirements.txt:

nginx
Copy
Edit
streamlit
yfinance
prophet
matplotlib
scikit-learn
pandas
numpy
🙌 Acknowledgements
Yahoo Finance – Stock data

Facebook Prophet – Time series forecasting

Streamlit – Dashboard framework
