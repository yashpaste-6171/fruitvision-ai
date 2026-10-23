
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="FruitVision AI",
    page_icon="🍎",
    layout="centered"
)

st.title("🍎 FruitVision AI")
st.write("Upload a fruit image to predict its class.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "fruit_classifier_final.keras"
    )

with open("fruit_class_names.json", "r") as f:
    class_names = json.load(f)

model = load_model()

uploaded_file = st.file_uploader(
    "Choose a fruit image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image",
             use_container_width=True)

    if st.button("Classify Fruit"):
        resized = image.resize((224, 224))
        array = np.asarray(resized, dtype=np.float32)
        array = np.expand_dims(array, axis=0)

        predictions = model.predict(array, verbose=0)[0]
        predicted_index = int(np.argmax(predictions))

        st.success(
            f"Prediction: {class_names[predicted_index]}"
        )
        st.write(
            f"Confidence: "
            f"{predictions[predicted_index] * 100:.2f}%"
        )
