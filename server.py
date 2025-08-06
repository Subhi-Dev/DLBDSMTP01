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
      "temperature": 0.0,
      "humidity": 10.0,
      "sound_volume": 2.8
    }
    """
    if loaded_model is None:
        return jsonify({"error": "Model not loaded. Please run the training script first."}), 500

    try:
        #Request Data and Validation
        data = request.get_json()

        if not data:
            return jsonify({"error": "Invalid JSON input."}), 400

        temperature = data.get('temperature')
        humidity = data.get('humidity')
        sound_volume = data.get('sound_volume')

        if None in [temperature, humidity, sound_volume]:
            return jsonify({"error": "Missing one or more required features: temperature, humidity, sound_volume"}), 400

        #Pandas DataFrame
        new_data = pd.DataFrame([[temperature, humidity, sound_volume]],
                                columns=['temperature', 'humidity', 'sound_volume'])

        new_data_scaled = loaded_scaler.transform(new_data)

        prediction_encoded = loaded_model.predict(new_data_scaled)

        predicted_inspection_due = loaded_label_encoder.inverse_transform(prediction_encoded)

        return jsonify({"predicted_inspection_due": "Yes" if int(predicted_inspection_due[0]) == 1 else "No"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # This block will run the Flask app on a local server.
    # To run this, make sure your model files are already created.
    app.run(host='0.0.0.0', debug=True)
