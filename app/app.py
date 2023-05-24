import folium
from streamlit_folium import folium_static
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from models.model import predicted_mask, iou_helper, accuracy_helper


# Function to preprocess the input image
def preprocess_image(image):
    # Resize the image to the desired input shape (e.g., 5000x5000)
    resized_image = image.resize((5000, 5000))
    # Perform any other preprocessing steps if needed
    # ...
    return resized_image

# Function to predict the segmentation mask using trained UNet model
def predict_mask(image):
    # create model object
    model = model()
    # Perform any necessary preprocessing on the image
    preprocessed_image = preprocess_image(image)

    # Convert the image to an array and normalize the RGB values
    image_array = np.array(preprocessed_image) / 255.0

    # Use your trained UNet model to predict the segmentation mask
    # ...
    # Replace the following line with your actual prediction code
    #predicted_mask = predicted_mask()

    return predicted_mask

# Streamlit app
def main():
    st.set_page_config(
    page_title='Rooftop Segmentation',
    page_icon="graphics/tab_icon.jpg",
    layout='wide'
    )

    st.title("Rooftop Segmentation")

    # Upload image or select an area on OpenStreetMaps
    option = st.radio("Select input", ("Upload satellite image", "Select area on map"))

    if option == "Upload satellite image":
        # Upload image from test dataset
        uploaded_image = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png", "tif"])
        if uploaded_image is not None:
            # Display the uploaded image
            image = Image.open(uploaded_image)



            # Perform prediction and get the segmentation mask
            predicted_mask = predict_mask(image)

            # Calculate and display the metrics (IoU, accuracy)
            # ...
            # Replace the following lines with your actual metric calculations
            iou = iou_helper
            accuracy = accuracy_helper
            st.subheader("Metrics")
            st.write("IoU:", iou)
            st.write("Accuracy:", accuracy)

            # Display the image with the mask overlaid on top
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
            ax1.imshow(image)
            ax1.set_title("Satellite image")
            ax1.axis("off")
            ax2.imshow(image)
            alpha_value = st.slider("Mask opacity", 0.0, 1.0, 0.3, step=0.1)
            ax2.imshow(predicted_mask, cmap="jet", alpha=alpha_value)
            ax2.set_title("Identified roofs")
            ax2.axis("off")

            # Show the figure in Streamlit
            st.pyplot(fig)

    else:
        st.subheader("Select an area on the map")

        # Define the initial center and zoom level for the map
        initial_coordinates = folium.Map(location=[29.750740286339706, -95.36208972613808], zoom_start=17, tiles='Stamen Terrain')

        # Display Maps satellite view
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Esri Satellite',
            overlay=False,
            control=True
        ).add_to(initial_coordinates)

        # Display the map in Streamlit
        folium_static(initial_coordinates, width=500, height=500)

        if st.button("Predict"):
            # Retrieve the visible area of the map
            bounds = initial_coordinates.get_bounds()
            if bounds is not None:
                lat_min, lon_min = bounds[0]
                lat_max, lon_max = bounds[1]
                if lat_min is not None and lon_min is not None and lat_max is not None and lon_max is not None:
                    # Create a folium map with the visible area
                    center_lat = (lat_min + lat_max) / 2
                    center_lon = (lon_min + lon_max) / 2
                    visible_area = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles='Stamen Terrain')
                    folium.Rectangle(
                        bounds=[[lat_min, lon_min], [lat_max, lon_max]],
                        color='red',
                        fill_opacity=0.2
                    ).add_to(visible_area)

                    # Convert the folium map to an image
                    image = visible_area._to_png()

                    # Display the image
                    st.image(image, caption="Visible area")

                    # Perform prediction and get the segmentation mask
                    predicted_mask = predict_mask(image)

                    # Calculate and display the metrics (IoU, accuracy)
                    iou = iou_helper
                    accuracy = accuracy_helper
                    st.subheader("Metrics")
                    st.write("IoU:", iou)
                    st.write("Accuracy:", accuracy)

                    # Display the satellite image with the mask overlaid on top
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                    ax1.imshow(image)
                    ax1.set_title("Satellite image")
                    ax1.axis("off")
                    alpha_value = st.slider("Mask opacity", 0.0, 1.0, 0.3, step=0.1)
                    ax2.imshow(image)
                    ax2.imshow(predicted_mask, cmap="jet", alpha=alpha_value)
                    ax2.set_title("Segmentation result")
                    ax2.axis("off")

                    # Show the figure in Streamlit
                    st.pyplot(fig)
        else:
            st.warning("Please zoom in to area of interest on the map before predicting.")


if __name__ == "__main__":
    main()
