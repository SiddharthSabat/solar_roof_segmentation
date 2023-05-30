import streamlit as st

st.subheader("Project Overview")
st.write("Utilize UNet CNN to accurately identify suitable roofs for solar panel installation. Enhance the adoption of renewable energy by automating the assessment process, reducing manual effort, and maximizing solar potential.")

st.subheader("Project Objectives")
st.write("The objective of this project is to create an automated system that utilizes a Deep Learning model architecture to accurately segment and identify roofs from aerial or satellite images. The system aims to support solar panel installation and provide valuable insights to organizations involved in solar energy. By uploading an aerial image or using location coordinates, companies can access the predicted mask of the image, thereby accelerating the growth of solar installations. This technology has the potential to contribute to the expansion and adoption of solar energy, enhance operational efficiency for businesses, enable informed decision-making, and promote sustainable development.")

col1, col2, col3, col4 = st.columns(4)
col1.image("images/chi1.jpg", caption="Chicago, IL")
col2.image("images/chi2.jpg", caption="Reference mask")
col3.image("images/kit1.jpg", caption="Kitsap County, WA")
col4.image("images/kit2.jpg", caption="Reference mask")

st.subheader("Dataset")
st.write("Dataset covers 810 km² (405 km² for training and 405 km² for testing).  Aerial orthorectified color imagery with a spatial resolution of 0.3 m. Ground truth data for two semantic classes: building and not building (publicly disclosed only for the training subset). The images cover dissimilar urban settlements, ranging from densely populated areas (e.g., San Francisco’s financial district) to alpine towns (e.g,. Lienz in Austrian Tyrol).")

st.subheader("Model Description")
st.write("The UNet architecture using VGG19, inspired by the encoder-decoder framework, features skip connections to preserve spatial information while capturing context. The backbone network, VGG19, serves as the encoder, extracting hierarchical features. The decoder, composed of upsample and convolutional layers, reconstructs high-resolution predictions for precise roof solar segmentation.")
