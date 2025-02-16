from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

# Load the TensorFlow model using tf.keras
try:
    model = tf.keras.models.load_model("deceptive_id_model.keras")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, Flask is working!"

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Please check the model file."}), 500
    
    try:
        # Get data from request
        data = request.json
        username = data.get('Username', "")
        followers = float(data.get('Followers', 0))
        posts = float(data.get('Posts', 0))
        sentiment = data.get('Sentiment', "Neutral")  # Default: Neutral
        reported = data.get('Reported', "No")  # Default: No
        bio_keywords = data.get('BioKeywords', "")  # Handle missing bio keywords

        # Convert input data to model-compatible format
        input_data = np.array([
            len(username), 
            followers, 
            posts, 
            len(bio_keywords), 
            1.0 if sentiment == "Positive" else 0.0, 
            1.0 if reported == "Yes" else 0.0
        ], dtype=np.float32)

        # Ensure correct shape for model input
        input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension
        
        # Make prediction
        prediction = model.predict(input_data)
        is_deceptive = 'Deceptive' if prediction[0][0] > 0.5 else 'Legit'

        return jsonify({
            'Username': username,
            'Prediction': is_deceptive
        })
    
    except Exception as e:
        return jsonify({"error": f"Invalid input or processing error: {e}"}), 400

if __name__ == '__main__':
    app.run(debug=True, port=8080)