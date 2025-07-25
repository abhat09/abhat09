import streamlit as st
import tensorflow as tf
import joblib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils.dino_features import extract_dino_feature

# Page setup
st.set_page_config(page_title="Neural Style Transfer & Fake Detection", layout="centered")
st.title("🧠 NEURAL STYLE TRANSFER & FAKE DETECTION")

# === Load models ===
@st.cache_resource
def load_style_model():
    return tf.saved_model.load("style_transfer_model")

@st.cache_resource
def load_svm_model():
    return joblib.load("model/svm_model.pkl")

style_model = load_style_model()
svm_model = load_svm_model()

# === Utility functions ===
def load_image(image_file, max_dim=512):
    img = Image.open(image_file).convert("RGB")
    img = img.resize((max_dim, max_dim))
    img = np.array(img) / 255.0
    return tf.expand_dims(tf.convert_to_tensor(img, dtype=tf.float32), axis=0)

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = tf.clip_by_value(tensor, 0, 255)
    tensor = tf.cast(tensor, tf.uint8)
    return Image.fromarray(tensor[0].numpy())

# === UI Layout ===
col1, col2 = st.columns(2)

# --- Stylize Image ---
with col1:
    st.subheader("Stylize Image")
    content_img = st.file_uploader("Drag and drop a content image", type=["jpg", "jpeg", "png"], key="content")

    if st.button("Stylize Image"):
        if content_img:
            st.image(content_img, caption="Original Image", use_container_width=True)

            content_tensor = load_image(content_img)

            st.info("🎨 Applying Cubism style...")
            with st.spinner("Stylizing..."):
                output = style_model(tf.constant(content_tensor), tf.constant(content_tensor))
                stylized_img = tensor_to_image(output[0])
                st.image(stylized_img, caption="Stylized Image", use_container_width=True)

# --- Fake Detection ---
with col2:
    st.subheader("Fake Detection")
    uploaded_file = st.file_uploader("Drag and drop an image", type=["jpg", "jpeg", "png"], key="auth")

    if st.button("Check Authenticity"):
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Artwork", use_container_width=True)

            st.info("🔍 Extracting features using DINOv2...")
            features = extract_dino_feature(image).reshape(1, -1)

            prob = svm_model.predict_proba(features)[0]
            pred = svm_model.predict(features)[0]

            label = "🧠 Human-made" if pred == 0 else "🤖 AI-generated"
            st.subheader(f"Prediction: {label}")
            st.write(f"Confidence: {prob[pred]*100:.2f}%")

            st.write("### 🔎 Prediction Confidence")
            classes = ["Human-made", "AI-generated"]
            colors = ["#4CAF50", "#FF5722"]

            fig, ax = plt.subplots()
            bars = ax.bar(classes, prob * 100, color=colors)
            ax.set_ylim([0, 100])
            ax.set_ylabel("Confidence (%)")
            ax.set_title("Prediction Confidence")

            for bar, p in zip(bars, prob):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, height + 1, f"{p*100:.1f}%", ha='center')

            st.pyplot(fig)

# Footer
st.markdown("---")
st.caption("Built using TensorFlow, DINOv2, scikit-learn, and Streamlit.")