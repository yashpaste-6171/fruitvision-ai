
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="FruitVision AI",
    page_icon="🍃",
    layout="wide"
)

# ---------- Custom design ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: linear-gradient(135deg, #e9fff0, #e8f8f5, #fff8e8);
    font-family: 'Inter', sans-serif;
    color: #111827;
}
.block-container {
    max-width: 1450px;
    padding-top: 1.2rem;
}
h1, h2, h3, p, label {
    color: #111827;
}
.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 15px 25px;
    border-radius: 50px;
    background: rgba(255,255,255,.9);
    box-shadow: 0 5px 25px #14532d18;
    margin-bottom: 25px;
}
.brand {
    font-size: 1.7rem;
    font-weight: 800;
}
.brand span {
    color: #16a34a;
}
.navlinks {
    font-size: .95rem;
    font-weight: 600;
    color: #334155;
}
.hero-title {
    font-size: clamp(2.5rem, 5vw, 4.5rem);
    line-height: 1.05;
    font-weight: 800;
    letter-spacing: -2px;
    margin-bottom: 12px;
}
.hero-title span {
    color: #16a34a;
}
.subtitle {
    font-size: 1.1rem;
    color: #526579 !important;
    line-height: 1.6;
}
.feature {
    text-align: center;
    padding: 17px 10px;
    margin-top: 14px;
    border-radius: 18px;
    background: rgba(255,255,255,.8);
    border: 1px solid white;
    box-shadow: 0 5px 15px #14532d0d;
}
.feature-icon {
    font-size: 1.6rem;
}
.feature strong {
    display: block;
    margin: 5px 0;
}
.feature small {
    color: #526579;
}
.panel {
    padding: 22px;
    border-radius: 23px;
    background: rgba(255,255,255,.88);
    border: 1px solid white;
    box-shadow: 0 7px 25px #14532d12;
}
.panel h3 {
    margin-top: 0;
}
.muted {
    color: #526579 !important;
}
.prediction {
    background: linear-gradient(90deg, #ffe1e6, #fff0e8);
    color: #be123c;
    border: 1px solid #fecdd3;
    border-radius: 14px;
    padding: 14px;
    font-size: 1.6rem;
    font-weight: 800;
    margin: 8px 0 15px;
}
.confidence {
    font-size: 2.3rem;
    font-weight: 800;
    color: #15803d;
}
.toprow {
    display: grid;
    grid-template-columns: 35px 1fr 75px;
    gap: 8px;
    align-items: center;
    padding: 11px 5px;
    border-bottom: 1px solid #e5eee7;
}
.rank {
    background: #e2e8f0;
    border-radius: 50%;
    width: 29px;
    height: 29px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
}
div.stButton > button {
    background: linear-gradient(90deg, #16a34a, #0d9747);
    color: white;
    font-size: 1.05rem;
    font-weight: 700;
    border: none;
    border-radius: 12px;
    padding: .75rem 1rem;
    width: 100%;
}
div.stButton > button:hover {
    background: #15803d;
    color: white;
    border: none;
}
.stFileUploader {
    background: #f5fff8;
    border: 1.5px dashed #7bd9a0;
    padding: 10px;
    border-radius: 15px;
}
.stFileUploader label,
.stFileUploader p,
.stFileUploader small {
    color: #334155 !important;
}
.footer {
    margin-top: 22px;
    padding: 22px 12px;
    border-radius: 22px;
    background: rgba(255,255,255,.88);
    text-align: center;
    color: #334155;
    border: 1px solid white;
}
@media (max-width: 750px) {
    .navbar {
        border-radius: 20px;
        flex-direction: column;
        gap: 8px;
    }
}
</style>
""", unsafe_allow_html=True)


# ---------- Model and labels ----------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "fruit_classifier_final.keras"
    )

with open("fruit_class_names.json", "r", encoding="utf-8") as f:
    class_names = json.load(f)


# ---------- Navigation ----------
st.markdown("""
<div class="navbar">
  <div class="brand">🍃 FruitVision <span>AI</span></div>
  <div class="navlinks">Home　 ·　 About　 ·　 How It Works　 ·　 Model</div>
</div>
""", unsafe_allow_html=True)


# ---------- Hero section ----------
hero_left, hero_right = st.columns([1, 1], gap="large")

with hero_left:
    st.markdown("""
    <div class="hero-title">
      FruitVision <span>AI</span>
    </div>
    <h2>AI-Powered Fruit Classification</h2>
    <p class="subtitle">
      Upload a fruit image and let our deep learning model
      identify its class. Explore intelligent fruit recognition
      with a simple, interactive experience.
    </p>
    """, unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""
        <div class="feature">
          <div class="feature-icon">⚡</div>
          <strong>AI Powered</strong>
          <small>EfficientNetB0</small>
        </div>""", unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class="feature">
          <div class="feature-icon">🍎</div>
          <strong>33 Classes</strong>
          <small>Fruit categories</small>
        </div>""", unsafe_allow_html=True)
    with f3:
        st.markdown("""
        <div class="feature">
          <div class="feature-icon">✨</div>
          <strong>Easy to Use</strong>
          <small>Upload and predict</small>
        </div>""", unsafe_allow_html=True)

with hero_right:
    st.image(
        "https://images.unsplash.com/photo-1610832958506-aa56368176cf"
        "?auto=format&fit=crop&w=1200&q=85",
        use_container_width=True
    )

st.write("")


# ---------- Upload panel ----------
left, right = st.columns([.92, 1.08], gap="large")

with left:
    st.markdown("""
    <div class="panel">
      <h3>☁️ Upload a Fruit Image</h3>
      <p class="muted">Supports JPG, JPEG and PNG formats.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a fruit image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    image = None

    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(
                image,
                caption="Your uploaded fruit",
                use_container_width=True
            )
        except Exception:
            st.error("Unable to open this image. Try another file.")

    classify_clicked = st.button(
        "✨  Classify Fruit",
        disabled=image is None
    )


# ---------- Prediction panel ----------
with right:
    st.markdown("""
    <div class="panel">
      <h3>📊 Prediction Result</h3>
      <p class="muted">Your AI results appear here.</p>
    </div>
    """, unsafe_allow_html=True)

    if classify_clicked and image is not None:
        try:
            with st.spinner("Analyzing your fruit..."):
                model = load_model()

                resized = image.resize((224, 224))
                array = np.asarray(resized, dtype=np.float32)
                array = np.expand_dims(array, axis=0)

                predictions = model.predict(array, verbose=0)[0]

            if len(predictions) != len(class_names):
                st.error(
                    "The model output count does not match the class "
                    "names. Please verify fruit_class_names.json."
                )
            else:
                top_indices = np.argsort(predictions)[::-1][:3]
                best_idx = int(top_indices[0])
                confidence = float(predictions[best_idx]) * 100

                st.markdown(
                    '<p class="muted">Predicted Class</p>'
                    f'<div class="prediction">🍓 '
                    f'{class_names[best_idx]}</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    '<p class="muted">Confidence</p>'
                    f'<div class="confidence">{confidence:.2f}%</div>',
                    unsafe_allow_html=True
                )

                st.progress(
                    min(max(confidence / 100, 0.0), 1.0)
                )

                st.markdown("### 🏅 Top 3 Predictions")

                for rank, idx in enumerate(top_indices, start=1):
                    score = float(predictions[idx]) * 100
                    label = class_names[int(idx)]

                    st.markdown(
                        f'<div class="toprow">'
                        f'<div class="rank">{rank}</div>'
                        f'<div><b>{label}</b></div>'
                        f'<div style="text-align:right">{score:.2f}%</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    st.progress(
                        min(max(score / 100, 0.0), 1.0)
                    )

        except Exception as exc:
            st.error("Prediction failed. Check the model and dependencies.")
            st.exception(exc)

    elif not uploaded_file:
        st.info("Upload a fruit image on the left to get started.")
    elif not classify_clicked:
        st.info("Click **Classify Fruit** to see the prediction.")


# ---------- Footer ----------
st.markdown("""
<div class="footer">
  🧠 <b>Deep Learning</b>　 ·　
  🗃️ <b>Fruit Dataset</b>　 ·　
  🎯 <b>Model Evaluation</b>　 ·　
  💻 <b>Streamlit Web App</b>
  <br><br>
  <span style="color:#64748b">
    🍃 FruitVision AI · Computer Vision Midterm Project
  </span>
</div>
""", unsafe_allow_html=True)
