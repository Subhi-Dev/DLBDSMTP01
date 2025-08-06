import pandas as pd
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

#Load the model
try:
    loaded_model = joblib.load('model/mlp_model.joblib')
    loaded_scaler = joblib.load('model/scaler.joblib')
    loaded_label_encoder = joblib.load('model/label_encoder.joblib')
    print("\nPre-trained model, scaler, and label encoder loaded successfully.")
except FileNotFoundError:
    print("\nError: Model files not found. Please run the training part of the script first to generate 'mlp_model.joblib', 'scaler.joblib', and 'label_encoder.joblib'.")
    loaded_model = None
    loaded_scaler = None
    loaded_label_encoder = None

@app.route('/predict', methods=['POST'])
def predict_weather_endpoint():
    """
    Flask endpoint to predict weather based on JSON input.
    Expected JSON format:
    {
      "precipitation": 0.0,
      "temp_max": 10.0,
      "temp_min": 2.8,
      "wind": 2.0
    }
    """
    if loaded_model is None:
        return jsonify({"error": "Model not loaded. Please run the training script first."}), 500

    try:
        #Request Data and Validation
        data = request.get_json()

        if not data:
            return jsonify({"error": "Invalid JSON input."}), 400

        precipitation = data.get('precipitation')
        temp_max = data.get('temp_max')
        temp_min = data.get('temp_min')
        wind = data.get('wind')

        if None in [precipitation, temp_max, temp_min, wind]:
            return jsonify({"error": "Missing one or more required features: precipitation, temp_max, temp_min, wind"}), 400

        #Pandas DataFrame
        new_data = pd.DataFrame([[precipitation, temp_max, temp_min, wind]],
                                columns=['precipitation', 'temp_max', 'temp_min', 'wind'])

        new_data_scaled = loaded_scaler.transform(new_data)

        prediction_encoded = loaded_model.predict(new_data_scaled)

        predicted_weather = loaded_label_encoder.inverse_transform(prediction_encoded)

        return jsonify({"predicted_weather": predicted_weather[0]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # This block will run the Flask app on a local server.
    # To run this, make sure your model files are already created.
    app.run(host='0.0.0.0', debug=True)
