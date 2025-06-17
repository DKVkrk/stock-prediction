import yfinance as yf
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import numpy as np
import pandas as pd
def fetch_stock_data(ticker):
    # Download historical stock data from Yahoo Finance
    df = yf.download(ticker, start='2015-01-01', end='2022-01-01')
    df.reset_index(inplace=True)  # Reset the index to get 'Date' as a column
    df = df[['Date', 'Close']]  # Keep only the 'Date' and 'Close' price columns
    df.columns = ['ds', 'y']  # Rename columns to fit Prophet's requirements
    df['ds'] = df['ds'].dt.tz_localize(None)

    # Adding moving average as a feature
    df['moving_avg'] = df['y'].rolling(window=5).mean().fillna(method='bfill')
    return df
def predict_stock_price(ticker, days):
    df = fetch_stock_data(ticker)  # Get historical stock data
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
    model.add_regressor('moving_avg')  # Add moving average as an external regressor
    model.fit(df)

    # Create future dates for predictions
    future = model.make_future_dataframe(periods=days)  # Generate future dates

    # Calculate moving average for future dates based on the last known values
    last_known_price = df['y'].iloc[-1]
    future['moving_avg'] = df['y'].iloc[-5:].mean()  # Use the last known moving average

    # Make predictions
    forecast = model.predict(future)  # Make predictions
    df.to_csv(f"{ticker}_historical_data.csv", index=False)  # Save historical data
    forecast.to_csv(f"{ticker}_predictions.csv", index=False)  # Save predictions
    merged_df = pd.merge(df, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='inner')

    # Save the merged DataFrame to a new CSV file
    merged_df.to_csv(f"{ticker}_merged_data.csv", index=False)


    # Get the last predicted stock price
    predicted_price = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days)
    return predicted_price, df, forecast
def calculate_accuracy(ticker):
    df = fetch_stock_data(ticker)  # Fetch historical data
    model = Prophet(changepoint_prior_scale=0.5, seasonality_prior_scale=10)
    model.add_regressor('moving_avg')  # Add moving average as an external regressor
    model.fit(df)  # Fit the model on historical data

    # Create a dataframe for predictions based on historical data
    forecast = model.predict(df)

    # Calculate Mean Absolute Percentage Error (MAPE)
    mape = mean_absolute_percentage_error(df['y'], forecast['yhat']) * 100  # Convert to percentage
    return mape, forecast

def calculate_mse_rmse(df, forecast):
    # Get actual and predicted values
    actual = df['y'].values
    predicted = forecast['yhat'].values[:len(actual)]  # Only compare within historical data length

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(actual, predicted)

    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)

    return mse, rmse

def plot_actual_vs_predicted(df, forecast):
    plt.figure(figsize=(14, 7))
    plt.plot(df['ds'], df['y'], label='Actual Price', color='blue')  # Actual prices
    plt.plot(forecast['ds'], forecast['yhat'], label='Predicted Price', color='orange')  # Predicted prices
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='orange', alpha=0.3)
    plt.title('Actual vs Predicted Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.axvline(x=df['ds'].iloc[-1], color='grey', linestyle='--', label='Last Training Date')  # Line indicating the last training date
    plt.legend()
    plt.grid()
    plt.show()
ticker_input = input("Enter the company ticker symbol (e.g., AAPL for Apple): ")
days_input = int(input("Enter the number of days to predict stock price: "))  # Get user input for prediction days

# Predict stock prices and display the results
try:
    predictions, historical_data, forecast = predict_stock_price(ticker_input, days_input)
    print(f"Predicted stock prices for {ticker_input} for the next {days_input} days:")
    print(predictions)
    latest_price = predictions['yhat'].iloc[-1]
    print(f"Latest predicted stock price: ${latest_price:.2f}")

    # Calculate and display model accuracy
    mape, forecast = calculate_accuracy(ticker_input)
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

    # Calculate MSE and RMSE
    mse, rmse = calculate_mse_rmse(historical_data, forecast)
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    # Plot actual vs predicted prices
    plot_actual_vs_predicted(historical_data, forecast)

except Exception as e:
    print(f"An error occurred: {e}")







