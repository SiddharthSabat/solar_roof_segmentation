
import streamlit as st

st.subheader("Performance Metrics")

st.write("**Accuracy**: Evaluate the overall pixel-level accuracy of the model's segmentations, indicating the proportion of correctly classified roof spaces. **IoU** (Intersection over Union): Quantify the overlap between predicted and ground truth segmentations, providing an assessment of segmentation quality.")

st.write("**Intersection over Union (IoU)**: The performance metric used for this project is Intersection over Union (IoU), it is used when calculating Mean average precision (mAP). It is a number from 0 to 1 that specifies the amount of overlap between the predicted and ground truth bounding box")

st.markdown("""
- an IoU of 0 means that there is no overlap between the boxes.
- an IoU of 1 means that the union of the boxes is the same as their overlap indicating that they are completely overlapping.
""")

col1 = st.columns(1)
col1.image("images/IoU_Metric.jpg", caption="IoU Metric Definition")
st.write("**Training Vs Validation IoU Trend From Model**)
col1 = st.columns(1)
col1.image("images/IOU_Metric_Trend.png", caption="IoU Metric Definition")


st.write("**Loss (Combination Loss):** The loss function used here is the combination of sigmoid cross-entropy loss and dice loss to form the total loss. The cross-entropy loss and the dice loss are added together to create a composite loss value.")
st.write("**Training Vs Validation Loss Trend From Model**)
col1 = st.columns(1)
col1.image("images/Loss_Trend.png", caption="IoU Metric Definition")

