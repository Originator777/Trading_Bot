import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from flask import Flask, request, jsonify
import alpaca_trade_api as tradeapi

# API creds
API_KEY = 'PK4QZNWN6CGRGD1U9CKI'
API_SECRET = 'Bo7RgzoiGOAs8TwnJ9BY6aGrhgVedT1GHE2hxWzv'
BASE_URL = 'https://paper-api.alpaca.markets'

# Initialize the Alpaca API client
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# Get account info
account = api.get_account()
print(account)

# Place order
api.submit_order(
    symbol = 'NVDA',
    qty = 1,
    side = 'buy',
    type = 'market',
    time_in_force = 'gtc'
)

# Get stock data
def get_stock_data(symbol, timeframe, start_date, end_date):
    # Retrieve historical price data
    barset = api.get_barset(symbol, timeframe, start=start_date, end=end_date)
    
    # Extract closing prices
    prices = []
    for bar in barset[symbol]:
        prices.append({
            'time': bar.t,
            'close': bar.c
        })
    
    return prices

if __name__ == "__main__":
    # Example usage
    symbol = 'AAPL'
    timeframe = 'day' 
    start_date = '2024-01-01'
    end_date = '2024-01-10'
    
    stock_data = get_stock_data(symbol, timeframe, start_date, end_date)
    print(stock_data)


# Load and preprocess the data
# Assuming having CSV file containing historical stock data of AMC
data = pd.read_csv('amc_stock_data.csv')

# Assuming CSV contains 'Date' and 'Close' columns
#'Close' prices as target variable
prices = data['Close'].values.reshape(-1, 1)

# Normalizing data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# Create sequences for LSTM model
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), 0])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

sequence_length = 50
X, y = create_sequences(scaled_prices, sequence_length)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the neural network model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Develop a Flask web application
app = Flask(__name__)

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Preprocess input data if necessary
    # Assuming you receive the last 50 prices
    input_prices = np.array(data['prices']).reshape(1, -1)
    scaled_input_prices = scaler.transform(input_prices)
    # Reshape data for prediction
    input_data = scaled_input_prices.reshape(1, sequence_length, 1)
    # Make predictions
    prediction = model.predict(input_data)
    # Inverse transform the prediction
    predicted_price = scaler.inverse_transform(prediction)
    return jsonify({'predicted_price': predicted_price[0][0]})

if __name__ == '__main__':
    app.run(debug=True)


