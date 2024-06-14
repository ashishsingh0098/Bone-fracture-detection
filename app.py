from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model
model = load_model('/home/ashish/VScode files/Python files/projects/DL project/Bone-fracture-detection/pre-trained-model/bone_fracture_dataset.h5')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']

    img = Image.open(io.BytesIO(file.read()))
    img = img.resize((150, 150)) 
    img_array = np.array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  

    # Make prediction
    prediction = model.predict(img_array)
    prediction = prediction[0][0] 

    return jsonify({'prediction': float(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
