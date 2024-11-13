from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import os
import numpy as np
from sklearn.decomposition import PCA
from skimage import io, color
from skimage.util import img_as_ubyte

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['COMPRESSED_FOLDER'] = 'compressed/'

@app.route('/')
def index():
    # Render index.html with no download path initially
    return render_template('index.html', download_path=None)

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No file uploaded", 400

    image = request.files['image']
    confidence = request.form.get('confidence')

    if image.filename == '':
        return "No file selected", 400

    # Save the uploaded image
    if image:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        image.save(image_path)
        
        np_value = float(confidence)
        compressed_image_filename = reduce_image(image_path, np_value)

        # Render index.html with the download path for the compressed image
        return render_template('index.html', download_path=compressed_image_filename, confidence=confidence)
    
    return "Upload failed", 500


def reduce_image(file_name, np_value):
    # Load the image and convert to grayscale
    image = io.imread(file_name)
    gray_image = color.rgb2gray(image)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=np_value)
    transformed_image = pca.fit_transform(gray_image)
    reconstructed_image = pca.inverse_transform(transformed_image)

    # Normalize and save the compressed image
    compressed_image_normalized = (reconstructed_image - reconstructed_image.min()) / (reconstructed_image.max() - reconstructed_image.min())
    compressed_image_uint8 = img_as_ubyte(compressed_image_normalized)
    
    # Save the compressed image
    compressed_image_filename = 'compressed_image.jpg'
    compressed_image_path = os.path.join(app.config['COMPRESSED_FOLDER'], compressed_image_filename)
    io.imsave(compressed_image_path, compressed_image_uint8)

    return compressed_image_filename

@app.route('/download/<filename>')
def download_file(filename):
    # Serve the compressed image file for download
    return send_from_directory(app.config['COMPRESSED_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    # Ensure upload and compressed directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['COMPRESSED_FOLDER'], exist_ok=True)
    
    app.run(debug=True)