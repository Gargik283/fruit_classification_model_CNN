import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from huggingface_hub import hf_hub_download

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Fresh vs Rotten",
    page_icon="🍎",
    layout="centered"
)

# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🍎 Fresh vs Rotten Image Classifier")

st.write(
    "Upload a fruit image and the CNN model will predict "
    "whether it is **Fresh** or **Rotten**."
)

# --------------------------------------------------
# Hugging Face Model Information
# --------------------------------------------------

REPO_ID = "Gargikundu22/fruit_classification_model"
MODEL_FILENAME = "fruit_classification_model.keras"

# --------------------------------------------------
# Download and Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILENAME,
        revision="main"
    )

    model = tf.keras.models.load_model(
        model_path
    )

    return model


# --------------------------------------------------
# Load Model
# --------------------------------------------------

try:

    model = load_model()

except Exception as e:

    st.error("❌ Unable to load the CNN model.")

    st.write(
        "Please make sure the following file exists "
        "inside your Hugging Face repository:"
    )

    st.code(MODEL_FILENAME)

    st.exception(e)

    st.stop()


# --------------------------------------------------
# Image Preprocessing
# --------------------------------------------------

def preprocess_image(image):

    # Resize to the same size used during training
    image = image.resize((224, 224))

    # Convert image to NumPy array
    image = np.array(
        image,
        dtype=np.float32
    )

    # Normalize pixel values
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
    "📤 Choose a fruit image",
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

    # Open uploaded image
    image = Image.open(
        uploaded_file
    ).convert("RGB")

    # Display image
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    st.write("")

    # Prediction button
    if st.button(
        "🔍 Predict",
        use_container_width=True
    ):

        with st.spinner(
            "Analyzing image..."
        ):

            # Preprocess image
            processed_image = preprocess_image(
                image
            )

            # Make prediction
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
        # Display Prediction
        # --------------------------------------------------

        st.subheader(
            f"Prediction: {label}"
        )

        st.write(
            f"Confidence Score: "
            f"**{confidence * 100:.2f}%**"
        )


        # --------------------------------------------------
        # Confidence Progress Bar
        # --------------------------------------------------

        st.progress(
            confidence
        )


        # --------------------------------------------------
        # Model Information
        # --------------------------------------------------

        with st.expander(
            "🔎 View Model Details"
        ):

            st.write(
                f"Raw sigmoid output: "
                f"`{rotten_probability:.4f}`"
            )

            st.write(
                "Input image size: `224 × 224`"
            )

            st.write(
                "Pixel normalization: `0–1`"
            )

            st.write(
                "Model type: `Convolutional Neural Network (CNN)`"
            )

            st.write(
                "Task: `Binary Image Classification`"
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
        This project uses a Convolutional Neural Network
        (CNN) built with TensorFlow/Keras to classify
        fruit images as Fresh or Rotten.
        """
    )

    st.divider()

    st.subheader(
        "🧠 Topics Covered"
    )

    st.write(
        """
        • Deep Learning

        • Convolutional Neural Networks (CNN)

        • Image Classification

        • Image Preprocessing

        • TensorFlow / Keras

        • NumPy

        • Model Evaluation

        • Streamlit Deployment
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
