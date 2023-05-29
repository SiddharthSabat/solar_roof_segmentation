from io import BytesIO
import folium as fl
import requests
from streamlit_folium import st_folium
from streamlit_folium import folium_static
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageGrab
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import gmaps
import streamlit.components.v1 as components

# Predict the mask using trained UNet model
def loss(y_true, y_pred):
    def dice_loss(y_true, y_pred):
        y_pred = tf.math.sigmoid(y_pred)
        numerator = 2 * tf.reduce_sum(y_true * y_pred)
        denominator = tf.reduce_sum(y_true + y_pred)
        return 1 - numerator / denominator
    y_true = tf.cast(y_true, tf.float32)
    cross_entropy_loss = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)
    total_loss = cross_entropy_loss + dice_loss(y_true, y_pred)
    return tf.reduce_mean(total_loss)

def iou_metric(y_true, y_pred):
    y_pred = tf.math.round(y_pred)
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    iou = intersection / (union + K.epsilon())
    return iou

def dice_loss(self, y_true, y_pred):
        smooth = 1e-5
        intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
        union = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3])
        dice_coeff = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice_coeff

model_paths = {
    'Baseline': 'models/baseline.h5',
    'UNet VGG19': 'models/unet_vgg18_model.h5',
    'UNet with Data Augmentation': 'models/Sid_Unet_Model_Train_Data_Aug_val.h5',
    'UNet without validation': 'models/Sid_Unet_Model_Train_2_Epoch.h5'
}

def predict_mask(image, selected_model):
    model_path = model_paths[selected_model]
    model = load_model(model_path, custom_objects={"loss": loss, "iou_metric": iou_metric, "dice_loss": dice_loss})

    image_size = (512, 512)
    image = np.array(image)
    image = cv2.resize(image, image_size)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0

    predicted_mask = model.predict(image)
    predicted_mask = np.squeeze(predicted_mask, axis=0)

    return predicted_mask

def model_summary(selected_model):
    model_path = model_paths[selected_model]
    model = load_model(model_path, custom_objects={"loss": loss, "iou_metric": iou_metric, "dice_loss": dice_loss})
    summary = []
    model.summary(print_fn=lambda x: summary.append(x))
    summary_str = "\n".join(summary)

    return summary_str

# Function to select area
def location_selector(location):
    coordinates = []

    if location == 'Austin, TX':
        coordinates = [30.274687230241007,-97.74036477412099]
    elif location == 'Cairo, Egypt':
        coordinates = [30.047601016747706,31.23850528556723]
    elif location == 'Houston, TX':
        coordinates = [29.750740286339706,-95.36208972613808]
    elif location == 'Mumbai, India':
        coordinates = [19.072743751435425,72.85868832704327]
    elif location == 'Oslo, Norway':
        coordinates = [59.912455005055214,10.744077188622049]
    elif location == 'Tyrol, Austria':
        coordinates = [47.282273863292524,11.516161884683973]
    else:
        coordinates = [29.750740286339706,-95.36208972613808]
    return coordinates

def load_api_key():
    with open("./google_maps_api.txt", "r") as file:
        api_key = file.read().strip()
    return api_key

# Streamlit app
def main():
    st.set_page_config(
        page_title='Rooftop Segmentation',
        page_icon="app/tab_icon.jpg",
        layout='wide'
    )

    st.subheader("Rooftop Segmentation")

    # Upload image or select an area on OpenStreetMaps
    option = st.radio("Select input", ("Upload satellite image", "Select area on map"))

    if option == "Upload satellite image":
        uploaded_image = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png", "tif"])
        if uploaded_image is not None:
            image = Image.open(uploaded_image)

            # Calculate and display the metrics
            with st.spinner(text="ML magic in progress..."):
                st.subheader("Metrics")
                col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
                selected_model = col1.selectbox('Prediction model', list(model_paths.keys()))
                predicted_mask = predict_mask(image, selected_model)
                resolution = col2.number_input('Image resolution, meters per px', 0.0, 100.0, 0.3)
                threshold = col3.number_input('Threshold', 0.0, 1.0, 0.5, step=0.1)
                col4.metric('Identified roof area, sq meters', int((np.count_nonzero(predicted_mask > threshold)) * resolution**2))

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Input image")
                    st.image(image, width=512)
                with col2:
                    st.subheader("Identified installation locations")
                    st.image(predicted_mask, width=512)

                if st.checkbox('Show model summary'):
                    st.write("Model Summary")
                    model_summary_text = model_summary(selected_model)
                    st.code(model_summary_text, language='python')


# Select on satellite map branch
    else:
        st.subheader("Select region")

        location = st.radio('Region', ['Austin, TX', 'Cairo, Egypt', 'Houston, TX', 'Mumbai, India', 'Oslo, Norway', 'Tyrol, Austria'])
        coordinates = location_selector(location)

        api_key = load_api_key()
        center = ','.join([str(coord) for coord in coordinates])
        zoom = 16
        map_url = f"https://maps.googleapis.com/maps/api/staticmap?center={center}&zoom={zoom}&size=1024x1024&maptype=satellite&key={api_key}"

        components.html(f'<img src="{map_url}">', height=600)

        response = requests.get(map_url)

        image = Image.open(BytesIO(response.content))

        image_variable = BytesIO()
        image = image.save(image_variable, format="PNG")

        image_path = "image.png"  # Specify the file path and name
        with open(image_path, "wb") as file:
            file.write(response.content)



        if st.button("Predict"):

            with st.spinner(text="ML magic in progress..."):
                st.subheader("Metrics")
                col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
                selected_model = col1.selectbox('Prediction model', list(model_paths.keys()))
                predicted_mask = predict_mask(image, selected_model)
                resolution = col2.number_input('Image resolution, meters per px', 0.0, 100.0, 0.3)
                threshold = col3.number_input('Threshold', 0.0, 1.0, 0.5, step=0.1)
                col4.metric('Identified roof area, sq meters', int((np.count_nonzero(predicted_mask > threshold)) * resolution**2))

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Input image")
                    st.image(image, width=512)
                with col2:
                    st.subheader("Identified installation locations")
                    st.image(predicted_mask, width=512)

                if st.checkbox('Show model summary'):
                    st.write("Model Summary")
                    model_summary_text = model_summary(selected_model)
                    st.code(model_summary_text, language='python')

if __name__ == "__main__":
    main()
