📈 Stock Price Prediction Dashboard
This is a Streamlit-based web application designed to forecast future stock prices. It uses the powerful Prophet forecasting library, developed by Facebook, to predict stock movements and provides a dashboard to visualize historical data, future predictions, and model performance metrics.

Overview
The application fetches historical stock data from Yahoo Finance, trains a Prophet model on the data, and generates a forecast for a specified number of future days. It displays the forecast in an interactive plot, along with key performance metrics like MAPE, RMSE, and MSE. Users can easily customize the stock ticker, date range, and prediction period through a clean sidebar interface.

Features
Customizable Ticker: Predict prices for any stock available on Yahoo Finance (e.g., AAPL, MSFT, GOOGL).

Flexible Date Range: Select a specific period of historical data for model training.

Adjustable Forecast Period: Choose how many days into the future you want to predict.

Interactive Visualization: A plot shows the actual historical prices, the predicted prices, and the confidence interval.

Performance Metrics: View key metrics (MAPE, MSE, RMSE) to evaluate the model's accuracy.

Trend and Seasonality Analysis: Visualize the underlying trend and seasonality components of the stock price.

Data Export: Download the generated forecast data as a CSV file.

Requirements
To run this application, you need to have Python and pip installed.

The required Python libraries are:

streamlit

yfinance

prophet

matplotlib

scikit-learn

numpy

pandas

How to Run
Clone or Download the Code: Get a copy of the Python script.

Install Required Libraries: Open your terminal or command prompt and run the following command to install all the necessary packages:

pip install streamlit yfinance prophet matplotlib scikit-learn numpy pandas

Note: Prophet has some dependencies, so if you encounter issues, refer to the official Prophet documentation for platform-specific installation instructions.

Run the Application: From the terminal in the same directory as your Python script, execute the following command:

streamlit run your_script_name.py

Replace your_script_name.py with the actual name of your file. This command will launch a local web server and open the dashboard in your default web browser.

Usage
Sidebar Settings: Use the sidebar on the left to configure your prediction.

Enter Ticker: Type a stock ticker symbol (e.g., AAPL for Apple) into the input field.

Select Dates: Choose the start and end dates for your training data.

Choose Days to Predict: Use the slider to select the number of days you want to forecast.

Run Prediction: Click the Run Prediction button to generate the forecast.

Explore Results: The main dashboard will update with the forecast plot, metrics, and other analysis. You can also check the boxes in the sidebar to show trend analysis or raw data tables.

Technologies Used
Streamlit: For creating the interactive web application.

yfinance: For fetching historical stock market data.

Prophet: A time-series forecasting library from Facebook.

Matplotlib: For data visualization.

Scikit-learn: For calculating model performance metrics.

Pandas & NumPy: For data manipulation and numerical operations.

Acknowledgments
This project was built using the power and flexibility of the open-source libraries listed above. A special thanks to the developers and communities behind them.
