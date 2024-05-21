from flask import Flask, request, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Linear Regression

df = pd.read_csv('data.csv')
df.dropna(inplace=True)

X = df[['sqft_lot', 'bedrooms', 'bathrooms']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Predict Function
def predict_price(sqft_lot, bedrooms, bathrooms):
    input_data = pd.DataFrame({
        'sqft_lot': [sqft_lot],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms]
    })
    predicted_price = model.predict(input_data)
    return predicted_price[0]

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for predicting house prices
@app.route('/predict', methods=['POST'])
def predict():
    sqft_lot = float(request.form['sqft_lot'])
    bedrooms = int(request.form['bedrooms'])
    bathrooms = float(request.form['bathrooms'])
    prediction = predict_price(sqft_lot, bedrooms, bathrooms)
    return render_template('index.html', prediction_text=f'The predicted price is ${prediction:.2f}')

if __name__ == '__main__':
    app.run(debug=True)
