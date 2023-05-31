import streamlit as st

st.subheader("Project Overview")
st.write("The project aims to develop an automated system using Deep Learning model architecture (UNet CNN and Vgg19) to enhance the adoption of renewable energy by automating the assessment process, reducing manual effort, and maximizing solar potential.")
st.write("The application can accurately segment and identify suitable roofs from satellite images for solar panel installation and to provide valuable insights to organizations involved in solar energy installation. The project would help to accelerate the growth of solar installations by accessing the predicted mask of the aerial images just by uploading the image of the target location or by using location coordinates.")

st.subheader("Business Application")
st.write("Calculating total area of the targeted location from the aerial image or from the map location.")

st.markdown("""
- An awesome tool to find out the estimated revenue for solar panel installation companies.
- The revenue can be estimated for the predicted solar roof area based on the input Electricity generation per square feet.
- It has the added feature for finding the suitability of whether the installation can be controlled using a threshold parameter.
""")

st.subheader("Industry Applications")

st.markdown("""
- Identify optimal areas on roofs for efficient solar panel installation can be used by Solar Installation companies.
- Construction companies can offer specialized services for integrating solar panels into roofs.
- It can support renewable energy initiatives and plan for solar panel deployment while doing the urban planning by the Govt agencies. This way the agencies can Promote clean energy solutions and reduce carbon footprints through solar energy adoption."]
""")

col1, col2, col3, col4 = st.columns(4)
col1.image("images/chi1.jpg", caption="Chicago, IL")
col2.image("images/chi2.jpg", caption="Reference mask")
col3.image("images/kit1.jpg", caption="Kitsap County, WA")
col4.image("images/kit2.jpg", caption="Reference mask")

st.subheader("Dataset")
st.write("Dataset covers 810 km² (405 km² for training and 405 km² for testing).  Aerial orthorectified color imagery with a spatial resolution of 0.3 m. Ground truth data for two semantic classes: building and not building (publicly disclosed only for the training subset). The images cover dissimilar urban settlements, ranging from densely populated areas (e.g., San Francisco’s financial district) to alpine towns (e.g,. Lienz in Austrian Tyrol).")

st.subheader("Model Description")
st.write("The UNet architecture using VGG19, inspired by the encoder-decoder framework, features skip connections to preserve spatial information while capturing context. The backbone network, VGG19, serves as the encoder, extracting hierarchical features. The decoder, composed of upsample and convolutional layers, reconstructs high-resolution predictions for precise roof solar segmentation.")
