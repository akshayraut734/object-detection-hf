# app.py
import os
import cv2
import numpy as np
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, redirect, url_for, abort
import base64
from io import BytesIO

app = Flask(__name__)

# Define category mapping
category_mapping = {
    "Apron": 0, "Baby Wear": 1, "Bedsheets": 2, "Blankets": 3, "Books": 4, "Cap": 5, "Chocolates": 6, "Clothes": 7,
    "Cooker": 8, "Decorative Items": 9, "Documents": 10, "Drugs": 11, "Dry Fruits": 12, "Explosives": 13, "Fabric": 14,
    "Gasoline": 15, "Gloves": 16, "Guns": 17, "Hair Dryer": 18, "Hair Straightner": 19, "Hairband Clips": 20,
    "Handicraft": 21, "Jackets": 22, "Jwellery": 23, "Mixer Grinder": 24, "Photo Frame": 25, "Rakhi": 26,
    "Ready to eat": 27, "Shoes": 28, "Snacks": 29, "Spices": 30, "Stationary Items": 31, "Sweets": 32, "Trimmer": 33,
    "Utensils": 34, "Watch": 35, "Wooden Items": 36
}

# Define dangerous categories
dangerous_categories = ["Explosives", "Guns", "revolver", "Drugs", "Gasoline"]

# Load the pretrained ViT model
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# Function to preprocess and predict class label of an image
def predict_image_class(image, processor, model):
    # Convert PIL image to NumPy array
    image_np = np.array(image)
    # Check if the image has 4 channels (RGBA), if so, convert it to RGB
    if image_np.shape[-1] == 4:
        image_np = image_np[:, :, :3]
    # Convert NumPy array to TensorFlow tensor
    image_tf = tf.convert_to_tensor(image_np, dtype=tf.float32)
    # Preprocess the image using ViTImageProcessor
    inputs = processor(images=image_tf, return_tensors="pt")
    # Forward pass through the ViT model
    outputs = model(**inputs)
    logits = outputs.logits
    # Post-process the output to get class label and confidence score
    predicted_class_idx = logits.argmax(-1).item()
    predicted_class_label = model.config.id2label[predicted_class_idx]
    confidence_score = logits.softmax(dim=-1).max().item()  # Confidence score
    
    return predicted_class_label, confidence_score

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the request contains the 'image_url' field
    if 'image_url' not in request.form:
        # Redirect to the home page if the 'image_url' field is not provided
        return redirect(url_for('home'))
    
    # Get the image URL from the POST request
    image_url = request.form['image_url']
    # Load and preprocess the image
    image = Image.open(requests.get(image_url, stream=True).raw)
    # Predict the class label and confidence score of the input image using the ViT model
    predicted_label, confidence_score = predict_image_class(image, processor, model)
    # Check if the predicted category is dangerous
    is_dangerous = predicted_label in dangerous_categories
    # Convert PIL image to base64 string for displaying in HTML
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    # Return the result as HTML
    return render_template('result.html', image=image_str, predicted_label=predicted_label, confidence_score=confidence_score, is_dangerous=is_dangerous)

@app.route('/exit', methods=['GET'])
def exit_application():
    # Terminate the Flask application
    os._exit(0)

if __name__ == '__main__':
    app.run(debug=True)
