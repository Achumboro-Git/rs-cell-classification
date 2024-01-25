import streamlit as st
from tensorflow.keras.preprocessing import image as tf_image
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Using the new caching mechanism
@st.cache_data
def load_model_cached():
    # Replace 'model.h5' with the path to your model file
    return load_model('enet96loss16_model.h5')

model = load_model_cached()

# Define the prediction function
def make_prediction(uploaded_image):
    # Preprocess the uploaded image
    img = uploaded_image.resize((290, 290))
    img_array = tf_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Model expects a batch of images
    img_array = img_array.astype('float32') / 255.0  # Normalize

    # Make the prediction
    prediction = model.predict(img_array)

    # Convert prediction to percentage and check the result
    prediction_percentage = prediction[0][0] * 100
    result = "Positive" if prediction_percentage >= 75 else "Negative"

    return {"score": prediction_percentage, "result": result}

# Streamlit application interface
st.title('Reed Sternberg Cell Classification AI Model for Hodgkins Lymphoma')

# File uploader
uploaded_file = st.file_uploader(label='Upload an image for analysis', type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # Convert the file to an image
    image = Image.open(uploaded_file).convert('RGB')

    # Display the uploaded image using Streamlit's built-in function
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Predicting the class of the image using the loaded model
    with st.spinner('Analyzing the image...'):
        prediction = make_prediction(image)
        st.success('Analysis complete')
        st.write(f'Prediction Score: {prediction["score"]:.2f}%')
        st.write(f'Result: {prediction["result"]}')

# Footer
st.markdown('© Andrews Kwadwo Owusu 2024')
