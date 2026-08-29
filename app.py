import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Fresh vs Rotten Classifier",
    page_icon="🍎",
    layout="centered"
)

# --------------------------------------------------
# Custom Styling
# --------------------------------------------------

st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #666;
        font-size: 16px;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-top: 20px;
        border: 1px solid #ddd;
    }

    .confidence {
        font-size: 18px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🍎 Fresh vs Rotten Image Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload a fruit image and let the CNN model classify it as Fresh or Rotten.'
    '</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# Load Trained CNN Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "fruit_fresh_rotten_classifier.keras"
    )


try:
    model = load_model()

except Exception as e:
    st.error("Unable to load the trained CNN model.")

    st.info(
        "Make sure the model file "
        "'fruit_fresh_rotten_classifier.keras' is available."
    )

    st.exception(e)
    st.stop()

# --------------------------------------------------
# Image Preprocessing
# --------------------------------------------------

def preprocess_image(image):

    # Resize image to model input size
    image = image.resize((224, 224))

    # Convert image to NumPy array
    image = np.array(image, dtype=np.float32)

    # Normalize pixel values
    image = image / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image


# --------------------------------------------------
# Image Upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "📤 Upload a fruit image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    st.write("")

    if st.button(
        "🔍 Predict",
        use_container_width=True
    ):

        with st.spinner("Analyzing image..."):

            processed_image = preprocess_image(image)

            prediction = model.predict(
                processed_image,
                verbose=0
            )

            # Sigmoid output:
            # 0 = Fresh
            # 1 = Rotten
            rotten_probability = float(
                prediction[0][0]
            )

        # --------------------------------------------------
        # Classification Result
        # --------------------------------------------------

        if rotten_probability > 0.5:

            label = "Rotten 🔴"
            confidence = rotten_probability

        else:

            label = "Fresh 🟢"
            confidence = 1 - rotten_probability

        # --------------------------------------------------
        # Display Result
        # --------------------------------------------------

        st.markdown(
            f"""
            <div class="result-box">

                <h2>{label}</h2>

                <p class="confidence">
                    Confidence: {confidence * 100:.2f}%
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        # --------------------------------------------------
        # Model Details
        # --------------------------------------------------

        with st.expander("View Model Output"):

            st.write(
                f"Raw sigmoid score: "
                f"`{rotten_probability:.4f}`"
            )

            st.write(
                "Input Image Size: `224 × 224`"
            )

            st.write(
                "Pixel Normalization: `0–1`"
            )

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("📌 About the Project")

    st.write(
        "This project uses a Convolutional Neural Network "
        "(CNN) built with TensorFlow/Keras to classify "
        "fruit images as Fresh or Rotten."
    )

    st.divider()

    st.subheader("🧠 Topics Covered")

    st.write(
        """
        - Deep Learning
        - Convolutional Neural Networks (CNN)
        - Image Classification
        - Image Preprocessing
        - TensorFlow / Keras
        - NumPy
        - Model Evaluation
        - Streamlit Deployment
        """
    )

    st.divider()

    st.subheader("⚙️ Model Information")

    st.write("**Model:** CNN")
    st.write("**Task:** Binary Classification")
    st.write("**Input Size:** 224 × 224")
    st.write("**Output:** Fresh / Rotten")
    st.write("**Activation:** Sigmoid")
    st.write("**Test Accuracy:** 95.40%")
