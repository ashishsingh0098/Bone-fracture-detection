from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import io



app = Flask(__name__)

# Load the trained model
model = load_model('/home/ashish/VScode files/Python files/projects/DL project/Bone-fracture-detection/pre-trained-model/bone_fracture_dataset.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        file = request.files['file']

        # Read and preprocess the image
        img = Image.open(io.BytesIO(file.read())).convert('RGB')  # Ensure the image is in RGB mode
        img = img.resize((150, 150))  # Adjust based on your model's input size
        img_array = np.array(img) / 255.0  # Normalize if needed
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 150, 150, 3)

        # Make prediction
        prediction = model.predict(img_array)
        prediction = prediction[0]  # Adjust if needed based on your model's output

        return jsonify({'prediction': float(prediction)})

if __name__ == '__main__':
    app.run(debug=True)

    # http://127.0.0.1:5000/predict