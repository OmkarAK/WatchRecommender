import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import io

# Load the embeddings and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Set up the pre-trained ResNet50 model for feature extraction
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Streamlit App Title
st.title('Watch Recommendation Based on Image')

# Function to extract features from the image
def feature_extraction(img, model):
    # Resize the image to the required shape (224, 224)
    img = img.resize((224, 224))  # Resize PIL Image to 224x224
    
    # Convert PIL Image to a numpy array
    img_array = image.img_to_array(img)
    
    # Expand dimensions to match the input shape for ResNet50 (1, 224, 224, 3)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocess the image for ResNet50
    preprocessed_img = preprocess_input(expanded_img_array)
    
    # Get feature vector from the model
    result = model.predict(preprocessed_img).flatten()
    
    # Normalize the result to unit length
    normalized_result = result / norm(result)
    
    return normalized_result

# Function to recommend similar watches
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# File uploader for the user to upload an image
uploaded_file = st.file_uploader("Upload Image of a Watch")
if uploaded_file is not None:
    try:
        # Open the uploaded file as a BytesIO object to handle stream-based image data
        img = Image.open(io.BytesIO(uploaded_file.read()))
    except Exception as e:
        st.error(f"Error opening image: {e}")
        st.stop()

    # Display the uploaded image
    st.image(img, caption="Uploaded Watch Image")

    # Extract features from the image
    features = feature_extraction(img, model)

    # Get the most similar watches based on the extracted features
    indices = recommend(features, feature_list)

    # Display similar watches in a row of columns
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.image(filenames[indices[0][0]], caption="Similar Watch 1")
    with col2:
        st.image(filenames[indices[0][1]], caption="Similar Watch 2")
    with col3:
        st.image(filenames[indices[0][2]], caption="Similar Watch 3")
    with col4:
        st.image(filenames[indices[0][3]], caption="Similar Watch 4")
    with col5:
        st.image(filenames[indices[0][4]], caption="Similar Watch 5")
