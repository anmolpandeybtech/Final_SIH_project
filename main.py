from keras.models import load_model
from flask import Flask, render_template, request, jsonify
from keras.preprocessing import image
import numpy as np
import json
from PIL import Image
import io
import os
import cv2

# Creating the app
app = Flask(__name__, static_url_path='/static')

# Loading the model
model = load_model("final_skin_disease_model.h5")



# Define the route for the home page
@app.route('/')
def home():
    # Add logic here to render the home page HTML template
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the request
    file = request.files['file']

    # Open the image using PIL
    test_image = Image.open(file)
    
    # Load and preprocess the image
    test_image = test_image.resize((224, 224))
    test_image = image.img_to_array(test_image)
    test_image /= 255.0
    test_image = np.expand_dims(test_image, axis=0)
    
    # Make predictions using the loaded model
    predictions = model.predict(test_image)
    predicted_class_index = np.argmax(predictions)
    
    class_labels = ["Enfeksiyonel", "Ekzama", "Acne", "Pigment", "Benign", "Malign"]
    predicted_class_label = class_labels[predicted_class_index]

    # Set probability threshold
    threshold = 0.6

    # Check if probability is above threshold
    if predictions[0, predicted_class_index] < threshold:
        return render_template('error.html', error='Inconclusive result. Please consult a healthcare professional for an accurate diagnosis')
    
    # Render the results page with the prediction
    return render_template('results.html', prediction=predicted_class_label)

def allowed_file(filename):
    # Add logic to check if the filename has an allowed extension (e.g., .jpg, .jpeg, .png)
    allowed_extensions = {'jpg', 'jpeg', 'png'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# Run the application   
if __name__ == '__main__':
    app.run(debug=True)
