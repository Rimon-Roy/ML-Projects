from flask import Flask, request, render_template, send_file
import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import io

app = Flask(__name__)

# Function to apply K-Means Clustering
def compress_image(image, k):
    # Convert PIL Image to NumPy array
    image_np = np.array(image)
    
    # Reshape the image to a 2D array of pixels
    pixels = image_np.reshape((-1, 3))

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    compressed_pixels = kmeans.cluster_centers_[kmeans.labels_]
    
    # Reshape the compressed pixels back to the original image dimensions
    compressed_pixels = compressed_pixels.reshape(image_np.shape).astype(np.uint8)
    
    # Convert NumPy array back to PIL Image
    compressed_image = Image.fromarray(compressed_pixels)
    return compressed_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    k = int(request.form.get('k', 16))  # Default K value is 16 if not provided

    # Open the uploaded image
    image = Image.open(file.stream)
    
    # Compress the image using K-Means clustering
    compressed_image = compress_image(image, k)

    # Save the compressed image to a BytesIO object
    img_byte_arr = io.BytesIO()
    compressed_image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)  # Go to the start of the BytesIO object

    # Return the compressed image as a downloadable file
    return send_file(img_byte_arr, mimetype='image/jpeg', download_name='compressed_image.jpg', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
