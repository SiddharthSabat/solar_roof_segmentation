import time
import folium
from streamlit_folium import folium_static
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageGrab
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from selenium import webdriver

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

#@st.cache_data
def predict_mask(image):
    model = load_model("models/unet_vgg18_model.h5", custom_objects={"loss": loss, "iou_metric": iou_metric})

    image_size = (512, 512)
    image = np.array(image)
    image = cv2.resize(image, image_size)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0

    predicted_mask = model.predict(image)
    predicted_mask = np.squeeze(predicted_mask, axis=0)

    return predicted_mask

# Function to select area
def location_selector(location):
    coordinates = []

    if location == 'Austin, TX':
        coordinates = [30.274687230241007, -97.74036477412099]
    elif location == 'Cairo, Egypt':
        coordinates = [30.047601016747706, 31.23850528556723]
    elif location == 'Houston, TX':
        coordinates = [29.750740286339706, -95.36208972613808]
    elif location == 'Mumbai, India':
        coordinates = [19.072743751435425, 72.85868832704327]
    elif location == 'Oslo, Norway':
        coordinates = [59.912455005055214, 10.744077188622049]
    elif location == 'Tyrol, Austria':
        coordinates = [47.282273863292524, 11.516161884683973]
    else:
        coordinates = [29.750740286339706, -95.36208972613808]
    return coordinates

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

            predicted_mask = predict_mask(image)

            # Calculate and display the metrics (IoU, accuracy)
            iou = 333
            accuracy = 555
            st.subheader("Metrics")
            col1, col2, col3, col4 = st.columns([3, 3, 3, 3])
            selected_model = col1.selectbox('Prediction model', ['UNet', 'UNet++', 'PSPNet', 'DeepLabV3+'])
            resolution = col2.number_input('Image resolution, meters per px', 0.0, 100.0, 0.3)
            col3.metric('Identified roof area, sq meters', int((np.count_nonzero(predicted_mask > 0.5)) * resolution**2))
            col4.metric('Roof area, sqm', 5_555.00)

            with st.spinner(text="ML magic in progress..."):
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Input image")
                    st.image(image, width=512)
                with col2:
                    st.subheader("Identified installation locations")
                    st.image(predicted_mask, width=512)

# Select on satellite map branch
    else:
        st.subheader("Select an area on the map")

        location = st.radio('Region', ['Austin, TX', 'Cairo, Egypt', 'Houston, TX', 'Mumbai, India', 'Oslo, Norway', 'Tyrol, Austria'])
        coordinates = location_selector(location)
        map = folium.Map(location=coordinates, zoom_start=16, tiles='Stamen Terrain')

       # Display satellite view
        folium.TileLayer(
           tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
           attr='Esri',
           name='Esri Satellite',
           overlay=False,
           control=True).add_to(map)

        # Display the map in Streamlit
        folium_static(map, width=500, height=500)

        if st.button("Predict"):

            html = map.get_root().render()
            fName='map.html'
            map.save(fName)
            mUrl= f'file:///map/{fName}'
            driver = webdriver.Chrome()
            driver.get(mUrl)
            time.sleep(2)
            driver.save_screenshot('output.png')
            driver.quit()







            # Perform prediction and get the segmentation mask
            predicted_mask = predict_mask(image)

            # Calculate and display the metrics (IoU, accuracy)
            iou = 333
            accuracy = 555
            st.subheader("Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric('IoU', iou)
            col2.metric('Accuracy', accuracy)
            col3.metric('Identified roof area, sqm', 7_777.00)
            col4.metric('Roof area, sqm', 5_555.00)
            #col5.metric('Latitude:', ul_latitude)
            #col6.metric('Longitude:', ul_longitude)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Input image")
                with st.spinner(text="ML magic in progress..."):
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
                    ax1.imshow(image)
                    ax1.axis("off")
            with col2:
                st.subheader("Potential installation locations")
                with st.spinner(text="ML magic in progress..."):
                    ax2.imshow(image)
                    alpha_value = st.slider("Mask opacity", 0.0, 1.0, 0.3, step=0.1)
                    ax2.imshow(predicted_mask, cmap="binary_r", alpha=alpha_value)
                    ax2.axis("off")

            # Show the figure in Streamlit
            st.pyplot(fig)

        else:
            st.warning("Please zoom in to the area of interest on the map before predicting.")

if __name__ == "__main__":
    main()
