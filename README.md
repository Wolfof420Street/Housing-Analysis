# House Price Prediction Web Application

## Overview
A Flask-based web application for predicting house prices using a pre-trained machine learning model.

## Prerequisites
- Python 3.8+
- Flask
- joblib
- NumPy

## Installation

1. Clone the repository
```bash
git clone https://github.com/Wolfof420Street/Housing-Analysis.git
cd <project-directory>
```

2. Create a virtual environment
```bash
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
```

3. Install dependencies
```bash
pip install flask joblib numpy scikit-learn
```

## Running the Application
```bash
python app.py
```
Access the application at `http://localhost:5000`

## API Endpoint
- `/predict` (POST): 
  - Accepts JSON with house features
  - Returns predicted house price

## Project Structure
- `app.py`: Flask application
- `house_price_model.pkl`: Trained machine learning model
- `templates/index.html`: Web interface

## Example Request
```json
{
  "features": [2000, 7, 2010, 8, 500, 2, 13, 150]
}
```

## Error Handling
Handles and returns JSON errors for invalid inputs

## License
MIT