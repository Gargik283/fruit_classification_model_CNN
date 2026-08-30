import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path
import requests

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
# Model Configuration
# --------------------------------------------------

MODEL_FILENAME = "fruit_classification_model.keras"

# IMPORTANT:
# Replace this with the direct Hugging Face download URL
# of your fruit_classification_model.keras file.
MODEL_URL = "YOUR_HUGGINGFACE_MODEL_URL"


# --------------------------------------------------
# Download Model
# --------------------------------------------------

def download_model():

    model_path = Path(MODEL_FILENAME)

    # If model already exists, don't download it again
    if model_path.exists():
        return model_path

    if MODEL_URL == "YOUR_HUGGINGFACE_MODEL_URL":
        st.error("Model URL has not been configured.")
        st.info(
            "Upload your .keras model to Hugging Face and replace "
            "MODEL_URL in app.py with its direct download URL."
        )
        st.stop()

    try:

        with st.spinner("Downloading trained CNN model..."):

            response = requests.get(
                MODEL_URL,
                stream=True,
                timeout=120
            )

            response.raise_for_status()

            with open(model_path, "wb") as file:

                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):

                    if chunk:
                        file.write(chunk)

        return model_path

    except Exception as e:

        st.error("Unable to download the trained model.")
        st.exception(e)
        st.stop()


# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():

    model_path = download_model()

    return tf.keras.models.load_model(
        model_path
    )


try:

    model = load_model()

except Exception as e:

    st.error("Unable to load the trained CNN model.")

    st.exception(e)

    st.stop()


# --------------------------------------------------
# Image Preprocessing
# --------------------------------------------------

def preprocess_image(image):

    # Resize image to the same size used during training
    image = image.resize((224, 224))

    # Convert image to NumPy array
    image = np.array(
        image,
        dtype=np.float32
    )

    # Normalize pixel values from 0-255 to 0-1
    image = image / 255.0

    # Add batch dimension
    image = np.expand_dims(
        image,
        axis=0
    )

    return image


# --------------------------------------------------
# Image Upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "📤 Upload a fruit image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

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

        with st.spinner(
            "Analyzing image..."
        ):

            processed_image = preprocess_image(
                image
            )

            prediction = model.predict(
                processed_image,
                verbose=0
            )

            # Sigmoid output
            #
            # 0 = Fresh
            # 1 = Rotten

            rotten_probability = float(
                prediction[0][0]
            )

        # --------------------------------------------------
        # Classification
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
        # Model Output
        # --------------------------------------------------

        with st.expander(
            "View Model Output"
        ):

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

    st.header(
        "📌 About the Project"
    )

    st.write(
        """
        This project uses a Convolutional Neural
        Network (CNN) built with TensorFlow/Keras
        to classify fruit images as Fresh or Rotten.
        """
    )

    st.divider()

    st.subheader(
        "🧠 Topics Covered"
    )

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

    st.subheader(
        "⚙️ Model Information"
    )

    st.write(
        "**Model:** CNN"
    )

    st.write(
        "**Task:** Binary Classification"
    )

    st.write(
        "**Input Size:** 224 × 224"
    )

    st.write(
        "**Output:** Fresh / Rotten"
    )

    st.write(
        "**Activation:** Sigmoid"
    )

    st.write(
        "**Test Accuracy:** 95.40%"
    )
