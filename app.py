import pandas as pd
from flask import Flask, render_template, request
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

app = Flask(__name__)

# Sample data for training the model (replace with your actual monthly data)
data = {
    'Month': pd.date_range(start='2021-01-01', periods=12, freq='M'),
    'Beans Sold': [1000, 1200, 1500, 1300, 1600, 1700, 1900, 1800, 2000, 2100, 2300, 2500]
}
df = pd.DataFrame(data)

# Extract month number and create 'Month_Num'
df['Month_Num'] = df['Month'].dt.month
df['Year'] = df['Month'].dt.year

# Create a 'Holiday' feature (for simplicity, just an example)
holidays = [1, 7, 12]  # January, July, December
df['Holiday'] = df['Month_Num'].isin(holidays).astype(int)  # 1 for holiday month, 0 for non-holiday

# Prepare features and target variable
X = df[['Month_Num', 'Holiday']]
y = df['Beans Sold'].values

# Normalize the target variable for better performance
y = (y - np.mean(y)) / np.std(y)

# Create a neural network model
model = keras.Sequential([
    layers.Input(shape=(X.shape[1],)),  # Input layer
    layers.Dense(64, activation='relu'),  # Hidden layer with 64 neurons
    layers.Dense(32, activation='relu'),  # Hidden layer with 32 neurons
    layers.Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=100, batch_size=4, verbose=0)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        month_str = request.form['month']
        month_num = pd.to_datetime(month_str).month
        is_holiday = int(month_num in holidays)

        # Make prediction
        input_data = np.array([[month_num, is_holiday]])
        prediction = model.predict(input_data)[0][0]  # Output from model

        # Denormalize the prediction
        prediction = prediction * np.std(y) + np.mean(y)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
