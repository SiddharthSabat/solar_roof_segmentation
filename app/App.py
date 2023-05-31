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
    'UNet with Patch and Data Aug': 'models/Changed_IOU_Epoch_12_Class_30-May.h5'
}

def patch_single_test_image(input_image_data, model, patch_size, overlap):
    # Load the image data using OpenCV
    #input_image = cv2.imdecode(np.frombuffer(input_image_data, np.uint8), cv2.IMREAD_COLOR)
    input_image_data = np.array(input_image_data)
    if input_image_data is None:
        print("Error loading the image.")
        return None

    # Get the image properties
    height, width, _ = input_image_data.shape

    # Calculate the number of patches in each dimension
    num_patches_x = (width - overlap) // (patch_size - overlap)
    num_patches_y = (height - overlap) // (patch_size - overlap)

    predicted_masks = []

    for i in range(num_patches_x):
        for j in range(num_patches_y):
            x_start = i * (patch_size - overlap)
            y_start = j * (patch_size - overlap)
            x_end = x_start + patch_size
            y_end = y_start + patch_size

            patch_data = input_image_data[y_start:y_end, x_start:x_end, :]

            # Normalize the patch data
            patch_data = patch_data / 255.0

            # Reshape the patch to match the model's input shape
            patch_data = np.expand_dims(patch_data, axis=0)

            # Predict the patch using the model
            predicted_patch = model.predict(patch_data)
            predicted_masks.append(predicted_patch)

    # Combine the predicted masks into a single array
    predicted_masks = np.concatenate(predicted_masks, axis=0)

    # Determine the dimensions of the final combined image
    combined_width = num_patches_x * patch_size
    combined_height = num_patches_y * patch_size

    # Create an empty array to store the combined image
    combined_image = np.zeros((combined_height, combined_width, 1))

    # Iterate over the predicted masks and place each patch in the corresponding location in the combined image
    for i in range(num_patches_x):
        for j in range(num_patches_y):
            x_start = i * (patch_size - overlap)
            y_start = j * (patch_size - overlap)
            x_end = x_start + patch_size
            y_end = y_start + patch_size

            combined_image[y_start:y_end, x_start:x_end, :] = predicted_masks[i * num_patches_y + j]

    # Reshape the combined image array to the final dimensions
    combined_image = combined_image.reshape((combined_height, combined_width))

    return combined_image


#@st.cache(allow_output_mutation=True, suppress_st_warning=True, show_spinner=False)
#def predict_mask(image, selected_model):
#
#    model_path = model_paths[selected_model]
#    custom_objects={"loss": loss, "iou_metric": iou_metric, "dice_loss": dice_loss}
#    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
#    image = np.array(image)
#    image_size = (512, 512)
#    image = cv2.resize(image, image_size)
#    image = np.expand_dims(image, axis=0)
#    image = image / 255.0
#    predicted_mask = model.predict(image)
#    predicted_mask = np.squeeze(predicted_mask, axis=0)
#
#    return predicted_mask

def predict_mask(image, selected_model):

    model_path = model_paths[selected_model]
    custom_objects={"loss": loss, "iou_metric": iou_metric, "dice_loss": dice_loss}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

    if model_path == 'models/Changed_IOU_Epoch_12_Class_30-May.h5':
          # Set the patch size and overlap
        patch_size = 512  # Size of each patch
        overlap = 0    # Overlap between patches

        input_image_data = image

        predicted_mask = patch_single_test_image(input_image_data, model, patch_size, overlap)

    else:

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
        coordinates = [lon, lat]
    return coordinates

api_key = st.secrets.secret.api_key

# Streamlit app

st.set_page_config(
        page_title='Roof Segmentation',
        page_icon="app/tab_icon.jpg",
        layout='wide'
    )

# Sidebar controls
st.sidebar.write("**Settings**")
selected_model = st.sidebar.selectbox('Prediction model', list(model_paths.keys()))
resolution = st.sidebar.number_input('Map resolution, m/px', 0.0, 100.0, 0.3)
zoom = st.sidebar.number_input('Map zoom', 10, 20, 16)
threshold = st.sidebar.number_input('Roof suitability threshold', 0.0, 1.0, 0.5, step=0.1)
revenue_rate = st.sidebar.number_input('Generation capacity, $/sq meter/yr', 0, 10000, 1000)
lon = st.sidebar.number_input('Longitude', -180.0, 180.0, 0.0)
lat = st.sidebar.number_input('Latitude', -90.0, 90.0, 0.0)

def main():
    #st.image("app/logo.png", width=30)
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
                col1, col2 = st.columns([1, 1])

                predicted_mask = predict_mask(image, selected_model)
                roof_area = col1.metric('Identified roof area, sq.m', int((np.count_nonzero(predicted_mask > threshold)) * resolution**2))
                col2.metric('Estimated revenue, M$/yr', int((revenue_rate) * int((np.count_nonzero(predicted_mask > threshold)) * resolution**2) / 1000))

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

        center = ','.join([str(coord) for coord in coordinates])
        st.subheader("Metrics")
        col1, col2, col3 = st.columns([1, 1, 2])

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
            predicted_mask = predict_mask(image, selected_model)
            roof_area = col1.metric('Identified roof area, sq.m', int((np.count_nonzero(predicted_mask > threshold)) * resolution**2))
            col2.metric('Estimated revenue, M$/yr', int((revenue_rate) * int((np.count_nonzero(predicted_mask > threshold)) * resolution**2) / 1000))

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
