
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="FruitVision AI",
    page_icon="🍓",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f0fff4, #fff9ed);
}
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
}
.hero {
    padding: 28px;
    border-radius: 24px;
    background: linear-gradient(120deg, #dcfce7, #fff7ed);
    border: 1px solid #d1fae5;
    margin-bottom: 22px;
}
.hero h1 {
    color: #14532d;
    font-size: 3rem;
    margin-bottom: 5px;
}
.hero p {
    color: #456052;
    font-size: 1.1rem;
}
.panel {
    background: rgba(255,255,255,0.85);
    border: 1px solid #dceee2;
    padding: 22px;
    border-radius: 20px;
    box-shadow: 0 5px 20px rgba(30,80,50,0.06);
}
.result {
    background: #ecfdf3;
    padding: 20px;
    border-radius: 16px;
    border-left: 5px solid #16a34a;
}
div.stButton > button {
    background: #16a34a;
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.7rem 1rem;
    font-weight: 700;
    width: 100%;
}
div.stButton > button:hover {
    background: #15803d;
    color: white;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "fruit_classifier_final.keras"
    )

with open("fruit_class_names.json", "r") as f:
    class_names = json.load(f)

st.markdown("""
<div class="hero">
    <h1>🍃 FruitVision AI</h1>
    <h3>Smart fruit recognition, powered by deep learning</h3>
    <p>Upload a fruit photo and let AI identify its class.</p>
</div>
""", unsafe_allow_html=True)

a, b, c = st.columns(3)

with a:
    st.markdown(
        '<div class="panel">⚡ <b>AI Powered</b><br>'
        'EfficientNetB0 transfer learning</div>',
        unsafe_allow_html=True
    )
with b:
    st.markdown(
        '<div class="panel">🍎 <b>33 Classes</b><br>'
        'Recognize fruit categories in the model</div>',
        unsafe_allow_html=True
    )
with c:
    st.markdown(
        '<div class="panel">📊 <b>Top Predictions</b><br>'
        'View confidence scores for the leading classes</div>',
        unsafe_allow_html=True
    )

st.write("")

left, right = st.columns([1, 1.15], gap="large")

with left:
    st.markdown(
        '<div class="panel"><h3>📤 Upload a fruit</h3>'
        '<p>Choose a JPG, JPEG, or PNG image.</p></div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose your image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Your uploaded fruit",
                 use_container_width=True)

with right:
    st.markdown(
        '<div class="panel"><h3>✨ Prediction results</h3>'
        '<p>Your AI results will appear here.</p></div>',
        unsafe_allow_html=True
    )

    if uploaded_file and st.button("🍓 Classify Fruit"):
        with st.spinner("Analyzing your image..."):
            model = load_model()
            resized = image.resize((224, 224))
            array = np.asarray(resized, dtype=np.float32)
            array = np.expand_dims(array, axis=0)

            predictions = model.predict(array, verbose=0)[0]

        if len(predictions) != len(class_names):
            st.error(
                "The number of model outputs does not match "
                "the class-name list. Please check the label mapping."
            )
        else:
            top_indices = np.argsort(predictions)[::-1][:3]
            best = int(top_indices[0])
            confidence = float(predictions[best]) * 100

            st.markdown(
                f'<div class="result"><h2>🍏 '
                f'{class_names[best]}</h2>'
                f'<h3>{confidence:.2f}% confidence</h3>'
                f'</div>',
                unsafe_allow_html=True
            )

            st.subheader("🏆 Top 3 predictions")

            for rank, idx in enumerate(top_indices, start=1):
                score = float(predictions[idx]) * 100
                st.write(
                    f"**{rank}. {class_names[idx]}** — "
                    f"{score:.2f}%"
                )
                st.progress(min(max(score / 100, 0.0), 1.0))

st.divider()

st.markdown(
    "<div style='text-align:center;color:#64748b;'>"
    "🍃 <b>FruitVision AI</b> · Computer Vision Midterm Project"
    "<br>Built with Python, TensorFlow and Streamlit"
    "</div>",
    unsafe_allow_html=True
)

