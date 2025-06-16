from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
from sklearn import BaseEstimator, TransformerMixin
import os

# Custom transformer for the pipeline (if it was saved within the pipeline)
# class LogTransformer(BaseEstimator, TransformerMixin):
#     def fit(self, X, y=None):
#         return self

#     def transform(self, X, y=None):
#         return np.log(X + 1)

class LogTransformer(BaseEstimator, TransformerMixin):

    # fit
    def fit(self, X, y=None):
        # self.feature_names = list(X.columns)
        self.n_features_in = X.shape[1]
        return self

    # transformer
    def transform(self, X, y=None):
        assert self.n_features_in == X.shape[1]
        # Add 1 to avoid log(0) and -inf
        return np.log(X + 1)





# --- Feature Engineering Logic ---
# This function should be identical to the one used during model training.
def extract_new_features(df):
    df = df.copy()

    # 1. AGE_RISK_LEVEL from AGE
    def map_age_risk(age):
        if 16 <= age <= 25:
            return 'high_risk'
        elif 26 <= age <= 39:
            return 'medium_risk'
        elif age >= 40:
            return 'low_risk'
        return 'unknown'
    df['AGE_RISK_LEVEL'] = df['AGE'].apply(map_age_risk)

    # 2. DRIVING_EXPERIENCE to numeric (assuming the pipeline expects the numeric version)
    # The form now directly collects the string category, so we process it here.
    def experience_to_numeric(exp_str):
        if exp_str == '0-9y':
            return 4.5
        elif exp_str == '10-19y':
            return 14.5
        elif exp_str == '20-29y':
            return 24.5
        elif exp_str == '30y+':
            return 35.0
        return 0.0
    df['DRIVING_EXPERIENCE'] = df['DRIVING_EXPERIENCE'].apply(experience_to_numeric)

    # 3. TOTAL_VIOLATIONS
    df['TOTAL_VIOLATIONS'] = (
        df['SPEEDING_VIOLATIONS'] +
        df['DUIS'] +
        df['PAST_ACCIDENTS']
    )

    # 4. DAILY_MILEAGE_ESTIMATE
    # Ensure ANNUAL_MILEAGE is numeric before division
    df['ANNUAL_MILEAGE'] = pd.to_numeric(df['ANNUAL_MILEAGE'], errors='coerce').fillna(0)
    df['DAILY_MILEAGE_ESTIMATE'] = df['ANNUAL_MILEAGE'] / 365

    # 5. IS_FAMILY_DRIVER
    df['IS_FAMILY_DRIVER'] = ((df['MARRIED'] == 1) & (df['CHILDREN'] > 0)).astype(int)

    return df

app = Flask(__name__)

# Load the trained pipeline
try:
    model = joblib.load('voting_pipeline.pkl')
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: Model file 'voting_pipeline.pkl' not found. The app will not work without it.")
    model = None

# Define the features that will be collected from the HTML form
FORM_FEATURES = [
    'AGE', 'GENDER', 'RACE', 'DRIVING_EXPERIENCE', 'EDUCATION', 'INCOME',
    'CREDIT_SCORE', 'VEHICLE_OWNERSHIP', 'VEHICLE_YEAR', 'MARRIED', 'CHILDREN',
    'ANNUAL_MILEAGE', 'VEHICLE_TYPE', 'SPEEDING_VIOLATIONS', 'DUIS',
    'PAST_ACCIDENTS'
]

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    proba = None
    input_values = {feature: '' for feature in FORM_FEATURES}
    error_message = None

    if request.method == 'POST':
        if not model:
            error_message = "Model is not loaded. Cannot make predictions."
        else:
            try:
                # 1. Collect raw data from the form
                form_data = {feature: request.form.get(feature) for feature in FORM_FEATURES}
                input_values = form_data.copy()

                # 2. Convert to DataFrame
                df = pd.DataFrame([form_data])

                # 3. Coerce ONLY the truly numeric columns to numeric types.
                #    Leave categorical string columns (like GENDER) as objects/strings.
                numeric_cols = [
                    'AGE', 'CREDIT_SCORE', 'CHILDREN', 'ANNUAL_MILEAGE',
                    'SPEEDING_VIOLATIONS', 'DUIS', 'PAST_ACCIDENTS', 'MARRIED',
                    'VEHICLE_OWNERSHIP'
                ]
                for col in numeric_cols:
                    # Use errors='coerce' to turn any invalid input into NaN
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                # 4. Apply the SAME feature engineering as in training
                df_features = extract_new_features(df)

                # 5. Make prediction
                # The pipeline will now receive the correct data types for each column
                prediction = model.predict(df_features)[0]
                proba = model.predict_proba(df_features)[0][1]

            except Exception as e:
                error_message = f"An error occurred: {str(e)}"
                print(f"Prediction error: {e}")

    return render_template(
        "predict.html",
        prediction=prediction,
        proba=proba,
        features=FORM_FEATURES,
        input_values=input_values,
        error_message=error_message
    )


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for programmatic access"""
    if not model:
        return jsonify({"error": "Model not loaded"}), 500
        
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Create DataFrame from incoming JSON
        df = pd.DataFrame([data])
        
        # Apply feature engineering
        df_features = extract_new_features(df)

        # Predict
        prediction = int(model.predict(df_features)[0])
        proba = float(model.predict_proba(df_features)[0][1])
        
        return jsonify({
            "prediction": prediction,
            "probability": proba,
            "risk_level": "HIGH RISK" if prediction == 1 else "LOW RISK"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    Ensure templates directory exists
    if not os.path.exists('templates'):
        os.makedirs('templates')
    app.run(debug=True, host='0.0.0.0', port=5000)
