from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

# Load the trained model
# Make sure you've saved your model in house_price_prediction.py as 'house_price_model.pkl'
model = joblib.load('house_price_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)

        # Predict house price
        predicted_price = model.predict(features)[0]

        # Return the predicted price
        return jsonify({'predicted_price': round(predicted_price, 2)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
