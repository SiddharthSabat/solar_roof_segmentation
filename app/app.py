import folium
from streamlit_folium import folium_static
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageGrab
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

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
    'UNet VGG19': 'models/unet_vgg18_model.h5'
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

            # Check if map_bounds is already initialized in session_state
            if "map_bounds" not in st.session_state:
                # Initialize map_bounds with default values
                st.session_state.map_bounds = [[0, 0], [0, 0]]

            # Get the coordinates of the selected area
            ul_latitude = st.session_state.map_bounds[0][0]
            ul_longitude = st.session_state.map_bounds[0][1]
            lr_latitude = st.session_state.map_bounds[1][0]
            lr_longitude = st.session_state.map_bounds[1][1]

            # Create a folium map
            map = folium.Map(location=[ul_latitude, ul_longitude], zoom_start=16, tiles="Stamen Terrain")

            # Open and display the map image
            map_image = Image.open("map_image.png")
            st.image(map_image)

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
