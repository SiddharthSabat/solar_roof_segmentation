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
from streamlit_pages.streamlit_pages import MultiPage

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
    custom_objects={"loss": loss, "iou_metric": iou_metric, "dice_loss": dice_loss}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    image = np.array(image)
    image_size = (512, 512)
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
        coordinates = [29.755778813031128,-95.36844450152464]
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
st.set_page_config(
        page_title='Rooftop Segmentation',
        page_icon="app/tab_icon.jpg",
        layout='wide'
    )

st.sidebar.title("Rooftop Segmentation")
st.sidebar.subheader("Project Objectives")
st.sidebar.write("Utilize UNet CNN to accurately identify suitable roofs for solar panel installation.")
st.sidebar.write("Enhance the adoption of renewable energy by automating the assessment process, reducing manual effort, and maximizing solar potential.")
st.sidebar.subheader("Dataset")
st.sidebar.write("Coverage of 810 km² (405 km² for training and 405 km² for testing). Two classes: roof and not roof. The images cover dissimilar urban settlements, ranging from densely populated areas to alpine towns.")
st.sidebar.subheader("Model Description")
st.sidebar.write("The UNet architecture using VGG19, inspired by the encoder-decoder framework, features skip connections to preserve spatial information while capturing context. The backbone network, VGG19, serves as the encoder, extracting hierarchical features. The decoder, composed of upsample and convolutional layers, reconstructs high-resolution predictions for precise roof solar segmentation.")
st.sidebar.subheader("Performace Metrics")
st.sidebar.write("**Accuracy**: Evaluate the overall pixel-level accuracy of the model's segmentations, indicating the proportion of correctly classified roof spaces.")
st.sidebar.write("**IoU** (Intersection over Union): Quantify the overlap between predicted and ground truth segmentations, providing assessment of segmentation quality.")

def main():

    st.image('app/logo.png', width=100)
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
                col1, col2, col3 = st.columns([1, 1, 2])
                selected_model = col1.selectbox('Prediction model', list(model_paths.keys()))
                threshold = col1.number_input('Threshold', 0.0, 1.0, 0.5, step=0.1)
                predicted_mask = predict_mask(image, selected_model)
                resolution = col2.number_input('Image resolution, m/px', 0.0, 100.0, 0.3)
                revenue_rate = col2.number_input('Electricity generation, $/sq meter/yr', 0.0, 1000.0, 100.0)
                roof_area = col3.metric('Identified roof area, sq.m', int((np.count_nonzero(predicted_mask > threshold)) * resolution**2))
                col3.metric('Estimated revenue, M$/yr', int((revenue_rate) * int((np.count_nonzero(predicted_mask > threshold)) * resolution**2) / 1000))

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

        location = st.radio('Region', ['Cairo, Egypt', 'Houston, TX', 'Mumbai, India', 'Oslo, Norway'])
        coordinates = location_selector(location)
        api_key = load_api_key()
        center = ','.join([str(coord) for coord in coordinates])
        col1, col2, col3 = st.columns([1, 1, 2])
        zoom = col2.number_input('Map zoom', 10, 20, 16)
        map_url = f"https://maps.googleapis.com/maps/api/staticmap?center={center}&zoom={zoom}&size=530x530&maptype=satellite&key={api_key}"
        #components.html(f'<img src="{map_url}">', height=512)
        response = requests.get(map_url)
        image = Image.open(BytesIO(response.content))
        image_var = BytesIO()
        image = image.save(image_var, format="PNG")
        image_path = "image.png"
        with open(image_path, "wb") as file:
            file.write(response.content)
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = image.crop((0, 0, 512, 512))

        # Calculate and display the metrics
        with st.spinner(text="ML magic in progress..."):
            st.subheader("Metrics")
            #col1, col2, col3 = st.columns([1, 1, 2])
            selected_model = col1.selectbox('Prediction model', list(model_paths.keys()))
            threshold = col1.number_input('Threshold', 0.0, 1.0, 0.5, step=0.1)
            predicted_mask = predict_mask(image, selected_model)
            #zoom = col2.number_input('Map zoom', 10, 20, 16)
            resolution = col2.number_input('Image resolution, m/px', 0.0, 100.0, 0.3)
            revenue_rate = col2.number_input('Electricity generation, $/sq.m/yr', 0.0, 1000.0, 100.0)
            roof_area = col3.metric('Identified roof area, sq.m', int((np.count_nonzero(predicted_mask > threshold)) * resolution**2))
            col3.metric('Estimated revenue, M$/yr', int((revenue_rate) * int((np.count_nonzero(predicted_mask > threshold)) * resolution**2) / 1000))

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

def about():
    st.write("Welcome to about page")
    if st.button("Click about"):
        st.write("Welcome to About page")


def contact():
    st.write("Welcome to contact page")
    if st.button("Click Contact"):
        st.write("Welcome to contact page")


# call app class object
app = MultiPage()
# Add pages
app.add_page("Home",main)
app.add_page("About",about)
app.add_page("Contact",contact)
app.run()

if __name__ == "__main__":
    main()
