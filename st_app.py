import streamlit as st
import gmaps
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Function to preprocess the input image
def preprocess_image(image):
    # Resize the image to the desired input shape (e.g., 5000x5000)
    resized_image = image.resize((5000, 5000))
    # Perform any other preprocessing steps if needed
    # ...
    return resized_image

# Function to predict the segmentation mask using your trained UNet model
def predict_mask(image):
    # Perform any necessary preprocessing on the image
    preprocessed_image = preprocess_image(image)

    # Convert the image to an array and normalize the RGB values
    image_array = np.array(preprocessed_image) / 255.0

    # Use your trained UNet model to predict the segmentation mask
    # ...
    # Replace the following line with your actual prediction code
    predicted_mask = np.random.randint(0, 2, size=(5000, 5000))

    return predicted_mask

# Streamlit app code
def main():
    st.title("Rooftop Segmentation")

    # Upload image or select an area on Google Maps
    option = st.radio("Select Input", ("Upload Satellite Image", "Select Area on Map"))

    if option == "Upload Satellite Image":
        # Upload image from test dataset
        uploaded_image = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png", "tif"])
        if uploaded_image is not None:
            # Display the uploaded image
            image = Image.open(uploaded_image)
            #st.subheader("Uploaded Satellite Image")
            #st.image(image, caption="Uploaded Satellite Image", use_column_width=True)

            # Perform prediction and get the segmentation mask
            predicted_mask = predict_mask(image)

            # Calculate and display the metrics (IoU, accuracy)
            # ...
            # Replace the following lines with your actual metric calculations
            iou = np.random.rand()
            accuracy = np.random.rand()
            st.subheader("Metrics")
            st.write("IoU:", iou)
            st.write("Accuracy:", accuracy)

            # Display the image with the mask overlayed on top
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
            ax1.imshow(image)
            ax1.set_title("Satellite Image")
            ax1.axis("off")
            ax2.imshow(image)
            ax2.imshow(predicted_mask, cmap="jet", alpha=0.3)
            ax2.set_title("Identified Roofs")
            ax2.axis("off")

            # Show the figure in Streamlit
            st.pyplot(fig)

    else:
        # Get user-selected area on the map
        st.subheader("Select an area on the map")

        # Set up Google Maps
        gmaps.configure(api_key="AIzaSyA91FHeKwuDKwa_aXQ5t0QuaymF38Cbsto")  # Replace with your Google Maps API key

        # Define the initial center and zoom level for the map
        initial_coordinates = (37.7749, -122.4194)  # San Francisco coordinates
        zoom_level = 15

        # Display the Google Maps satellite view
        fig = gmaps.figure(center=initial_coordinates, zoom_level=zoom_level, map_type='SATELLITE')

        drawing = gmaps.drawing_layer()
        fig.add_layer(drawing)

        # Display the image in Streamlit
        #st.subheader("Google Maps Satellite View")
        st.write(fig)

        if st.button("Predict"):
            # Retrieve the selected polygon from the map
            if len(drawing.features) > 0:
                polygon = drawing.features[0].geometry

                # Fetch satellite image for the selected area based on the polygon coordinates
                # ...

                # Perform prediction and get the segmentation mask
                predicted_mask = predict_mask(image)

                # Calculate and display the metrics (IoU, accuracy)
                # ...
                # Replace the following lines with your actual metric calculations
                iou = np.random.rand()
                accuracy = np.random.rand()
                st.subheader("Metrics")
                st.write("IoU:", iou)
                st.write("Accuracy:", accuracy)

                # Display the satellite image with the mask overlayed on top
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                ax1.imshow(image)
                ax1.set_title("Satellite Image")
                ax1.axis("off")
                ax2.imshow(image)
                ax2.imshow(predicted_mask, cmap="jet", alpha=0.3)
                ax2.set_title("Segmentation Result")
                ax2.axis("off")

                # Show the figure in Streamlit
                st.pyplot(fig)
            else:
                st.warning("Please draw a polygon on the map before predicting.")



if __name__ == "__main__":
    main()
