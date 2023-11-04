from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score
import yfinance as yf
import streamlit as st
from datetime import date
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pickle
import matplotlib.pyplot as plt

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")
# cryptos = [
#
cryptos = ["BTC-USD", "ETH-USD", "XRP-USD"]


def load_data():
    cryptos =  ["BTC-USD", "ETH-USD", "XRP-USD"]

    start_date_unix = 1627776000  # Unix timestamp (August 1, 2023)
    end_date = date.today()
    end_date_unix = int(datetime(end_date.year, end_date.month, end_date.day).timestamp())
    # Fetch data from Yahoo Finance using yfinance
    crypto_data = {}
    for crypto in cryptos:
        crypto_df = yf.download(crypto, start=start_date_unix, end=end_date_unix, interval="1d")
        crypto_data[crypto] = crypto_df
    return crypto_data


@st.cache_data(experimental_allow_widgets=True)
def all():
    selected_coins = st.selectbox("Select dataset for prediction", cryptos)
    crypto_data = load_data()
    st.write("loading the coin data")

    if st.button("show data end lines"):
        st.subheader('Raw data')
        st.write(crypto_data[selected_coins].tail())
    future_days = st.slider("Number of days to forecast", 1, 30)
    # period = n_months * 15
    fp = st.button("Cryptocurrency Price Forecast")
    if fp:
        st.title("Cryptocurrency Price Forecast")

    err = st.button("system accuracy")
    xb = st.button("Find Best Profitable Coin")

    return selected_coins, future_days, xb, fp, err



#st.write("NETWORK ERROR TRY AGAIN slowly after loading the coin")
selected_coins, future_days, xb, fp, err=all()

sequence_length = 100

scalers = {}
closing_prices_scaled = {}
closing_prices_dict = {}
for crypto in cryptos:
    if crypto in crypto_data:

        crypto_df = crypto_data[crypto]
        print(f"{crypto} {crypto_df.tail(70)}")
        timestamps = np.array(crypto_df.index)
        closing_prices = np.array(crypto_df["Close"])

        if closing_prices.shape[0] > 0:
            scaler = MinMaxScaler()
            closing_prices_dict[crypto] = closing_prices
            closing_prices_scaled[crypto] = scaler.fit_transform(closing_prices.reshape(-1, 1))
            scalers[crypto] = scaler
# create x sequences and y values
# create x sequences and y values
def create_dataset(closing_prices, time_step=1):
    dataX, dataY = [], []
    for i in range(len(closing_prices) - time_step - 1):
        a = closing_prices[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(closing_prices[i + time_step])
    return np.array(dataX), np.array(dataY)

# Define the sequence length you want
X_data_sequences = {}
y_data_values = {}

for crypto in cryptos:
    if crypto in closing_prices_scaled:
        closing_prices = closing_prices_scaled[crypto]
        X_data_sequences[crypto], y_data_values[crypto] = create_dataset(closing_prices, time_step=sequence_length)
        print(y_data_values[crypto])
# Now you have X_data_sequences containing sequences of closing prices and y_data_values containing corresponding y values
# for crypto in cryptos:
#     if crypto in X_train_dict and crypto in y_train_dict:
#         print(f"{crypto}:")
#         print("X_train shape:", X_train_dict[crypto].shape)
#         print("y_train shape:", y_train_dict[crypto].shape)
## Create a function to split data without using train_test_split
def manual_split(data, val_ratio, test_ratio):
    total_samples = len(data)
    val_samples = int(total_samples * val_ratio)
    test_samples = int(total_samples * test_ratio)
    train_samples = total_samples - val_samples - test_samples
    train_data = data[:train_samples]
    val_data = data[train_samples:(train_samples + val_samples)]
    test_data = data[(train_samples + val_samples):]
    return train_data, val_data, test_data

# Set the ratios for validation and test sets
val_ratio = 0.1  # 10% for validation
test_ratio = 0.1  # 10% for test
# Creating dictionaries to store train, validation, and test datasets
X_train_dict = {}
X_val_dict = {}
X_test_dict = {}
y_train_dict = {}
y_val_dict = {}
y_test_dict = {}
for crypto in cryptos:
    if crypto in X_data_sequences and crypto in y_data_values:
        X_sequences = X_data_sequences[crypto]
        y_values = y_data_values[crypto]
        X_train, X_val, X_test = manual_split(X_sequences, val_ratio, test_ratio)
        y_train, y_val, y_test = manual_split(y_values, val_ratio, test_ratio)
        X_train_dict[crypto] = X_train
        X_val_dict[crypto] = X_val
        X_test_dict[crypto] = X_test
        y_train_dict[crypto] = y_train
        y_val_dict[crypto] = y_val
        y_test_dict[crypto] = y_test
# for crypto in cryptos:
#     print(f"{crypto}:")
#     print("X_train shape:", X_train_dict[crypto].shape)
#     print("X_val shape:", X_val_dict[crypto].shape)
#     print("X_test shape:", X_test_dict[crypto].shape)
#     print("y_train shape:", y_train_dict[crypto].shape)
#     print("y_val shape:", y_val_dict[crypto].shape)
#     print("y_test shape:", y_test_dict[crypto].shape)
# Define the sequence length you want
X_train_reshaped = {}
X_val_reshaped = {}
X_test_reshaped = {}
y_train_reshaped = {}
y_val_reshaped = {}
y_test_reshaped = {}
for crypto in cryptos:
    if crypto in X_train_dict and crypto in X_val_dict and crypto in X_test_dict and crypto in y_train_dict and crypto in y_val_dict and crypto in y_test_dict:
        # Reshape X_train
        X_train_reshaped[crypto] = X_train_dict[crypto].reshape(X_train_dict[crypto].shape[0], sequence_length, 1)
        # Reshape X_val
        X_val_reshaped[crypto] = X_val_dict[crypto].reshape(X_val_dict[crypto].shape[0], sequence_length, 1)
        # Reshape X_test
        X_test_reshaped[crypto] = X_test_dict[crypto].reshape(X_test_dict[crypto].shape[0], sequence_length, 1)
        # Reshape y_train
        y_train_reshaped[crypto] = y_train_dict[crypto].reshape(y_train_dict[crypto].shape[0], 1)
        # Reshape y_val
        y_val_reshaped[crypto] = y_val_dict[crypto].reshape(y_val_dict[crypto].shape[0], 1)
        # Reshape y_test
        y_test_reshaped[crypto] = y_test_dict[crypto].reshape(y_test_dict[crypto].shape[0], 1)

# Print the reshaped array shapes for one cryptocurrency (e.g., "BTC-USD")
# print("X_train_reshaped shape:", X_train_reshaped[example_crypto].shape)
# print("X_val_reshaped shape:", X_val_reshaped[example_crypto].shape)
# print("X_test_reshaped shape:", X_test_reshaped[example_crypto].shape)
# print("y_train_reshaped shape:", y_train_reshaped[example_crypto].shape)
# print("y_val_reshaped shape:", y_val_reshaped[example_crypto].shape)
# print("y_test_reshaped shape:", y_test_reshaped[example_crypto].shape)
# #st.write(X_train_reshaped[selected_coins])
@st.cache_data
def train_models(cryptos, X_train_reshaped, y_train_reshaped, batch_size=32, epochs=77):
    lstm_models = {}
    for crypto in cryptos:
        if crypto in X_train_reshaped and y_train_reshaped:
            model = Sequential()
            # model.add(LSTM(units=400, return_sequences=True, input_shape=(sequence_length, 1)))
            # model.add(LSTM(units=400, return_sequences=True, input_shape=(sequence_length, 1)))
            # model.add(LSTM(units=370, return_sequences=True, input_shape=(sequence_length, 1)))
            model.add(LSTM(units=300, return_sequences=True, input_shape=(sequence_length, 1)))
            model.add(LSTM(units=250, return_sequences=True, input_shape=(sequence_length, 1)))
            model.add(LSTM(units=150, return_sequences=True, input_shape=(sequence_length, 1)))
            model.add(LSTM(units=50))
            model.add(Dense(units=1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train_reshaped[crypto], y_train_reshaped[crypto], epochs=epochs, batch_size=batch_size)
            lstm_models[crypto] = model
    return lstm_models

#st.file_uploader
mb=st.button("train")
#if mb:
lstm_models = train_models(cryptos, X_train_reshaped, y_train_reshaped)
   # save the lstm_models as a pickle file

#     model_pkl_file = "New_lstm_models.pkl"
#     with open(model_pkl_file, 'wb') as file:
#         pickle.dump(lstm_models, file)
# model_pkl_file= "New_lstm_models.pkl"
# if mb:
#     model_pkl_file = "New_lstm_models.pkl"
# with open(model_pkl_file, 'rb') as file:
#     lstm_models = pickle.load(file)
    #return lstm_models
#if mb:
#saved_model(lstm_models)
from sklearn.metrics import r2_score
# load model from pickle file
@st.cache_data
def future_function():
    future_predictions = {}
    #if mb:
    for crypto in cryptos:
        last_sequence = X_test_dict[crypto][-1]
        future_pred = []
        for i in range(30):
            last_sequence = last_sequence.reshape(1, sequence_length, 1)
            next_price = lstm_models[crypto].predict(last_sequence, verbose=0)
            future_pred.append(next_price)
            next_price = next_price.reshape(1, 1, 1)
            last_sequence = np.concatenate((last_sequence[:, 1:, :], next_price), axis=1)
        future_pred = np.array(future_pred)
        future_pred = future_pred.reshape(1, -1)
        future_pred = scalers[crypto].inverse_transform(future_pred)
        future_predictions[crypto] = future_pred
    return future_predictions
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# for crypto, r2 in r2_scores.items():
#     st.write(f"R2 Score for {crypto}: {r2:.6f}")
# ... (Your existing code for loading and preprocessing data) ...
# Create a Streamlit app

# fullclose_past=crypto_data[close].tolist()
#fullclose_past.extend(price_list) #timestapms
# Display predicted price graphs for selected cryptocurrencies
future_predictions=future_function()
selected_coins, future_days, xb, fp, err = all()
if fp:
    price_array = future_predictions.get(selected_coins, None)
    if price_array is not None:
        st.subheader(f"Predicted Prices for {selected_coins} for the Next {future_days} Days")
        # Create a plot
        plt.figure(figsize=(10, 6))
        plt.plot(price_array[0,0:future_days], label="Predicted Prices", color='blue')
        plt.xlabel("Day")
        plt.ylabel("Price")
        plt.title(f"Predicted Prices for {selected_coins} for the Next {future_days} Days")
        plt.legend()
        plt.grid(True)
        # Display the plot in Streamlit
        st.pyplot(plt)
        # Display predicted prices for each day
        st.write("Predicted Prices:")
        for i, price in enumerate(price_array[0,0:future_days]):
            st.write(f"Day {i+1}: ${price:.6f}")
        st.write("---")  # Add a separator between cryptocurrency predictions
@st.cache_data
def calculate_potential_profit(prices):
    initial_price = prices[0]
    final_price = prices[-1]
    percent_change = ((final_price - initial_price) / initial_price) * 100
    return percent_change
st.title("Crypto Price Prediction and Profit Analysis")
# Add a slider to input the number of days for profit calculation
num_days = st.slider("Select the number of days for profit calculation", min_value=1, max_value=30, value=7)
# Create a button to find the best profitable coin
best_coin = None
if xb:
    st.subheader("Best Performing Cryptocurrency")
    max_profit = -float("inf")
    for crypto in cryptos:
        future_predictions = future_function()
        price_array = future_predictions.get(crypto, None)
        if price_array is not None:
            potential_profit = calculate_potential_profit(price_array[0, :num_days])
            st.write(f"Potential Profit for {crypto}: {potential_profit:.2f}%")
            # Update best coin if necessary
            if potential_profit > max_profit:
                best_coin = crypto
                max_profit = potential_profit
    st.write(f"The best cryptocurrency with the highest potential profit is: {best_coin}")
    st.write("---")
    # Display the best-performing cryptocurrency and its potential profit
    st.write(f"Potential Profit Percentage: {max_profit:.2f}%")
if err:
    st.title("system accuracy")
    absolute_mean_errors = {}
    r2_scores={}
    for crypto, model in lstm_models.items():
        if crypto in X_test_reshaped and crypto in scalers:
            y_pred = lstm_models[crypto].predict(X_test_reshaped[crypto])
            y_pred = scalers[crypto].inverse_transform(y_pred)
            y_actual = scalers[crypto].inverse_transform(y_test_reshaped[crypto])
            absolute_mean_error = mean_absolute_error(y_actual, y_pred)
            absolute_mean_errors[crypto] = absolute_mean_error
            r2 = r2_score(y_actual, y_pred)
            r2_scores[crypto] = r2
            st.write(f"R2 Score for {crypto}: {r2:.6f}")
            st.write(f"Absolute Mean Error for {crypto},: {absolute_mean_error}")
st.write("THANK YOU, HAPPY TRADING")
