# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from datetime import datetime

# Load your business data (replace 'your_data.csv' with your dataset)
data = pd.read_csv('your_data.csv')
data['Date'] = pd.to_datetime(data['Date'])  # Convert date column to datetime type
data.set_index('Date', inplace=True)  # Set date as index

# Visualize the data
plt.figure(figsize=(12, 6))
plt.plot(data)
plt.title('Business Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

# Check for stationarity (e.g., using the Dickey-Fuller test)
from statsmodels.tsa.stattools import adfuller

result = adfuller(data['Value'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])
if result[1] <= 0.05:
    print('Data is stationary.')
else:
    print('Data is not stationary. Apply differencing.')

# Differencing (if necessary)
# data_diff = data.diff().dropna()

# Fit ARIMA model
p, d, q = 1, 1, 1  # Example values for p, d, q (order of ARIMA model)
model = ARIMA(data, order=(p, d, q))
model_fit = model.fit(disp=0)

# Forecast future values
forecast_horizon = 10  # Adjust the number of future periods to forecast
forecast, stderr, conf_int = model_fit.forecast(steps=forecast_horizon)

# Generate future date index
future_dates = [data.index[-1] + pd.DateOffset(days=i) for i in range(1, forecast_horizon + 1)]

# Create a DataFrame for the forecast
forecast_df = pd.DataFrame({'Forecast': forecast}, index=future_dates)

# Plot the original data and the forecast
plt.figure(figsize=(12, 6))
plt.plot(data, label='Original Data')
plt.plot(forecast_df, label='Forecast', color='red')
plt.title('Business Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

# Print the forecasted values
print(forecast_df)
