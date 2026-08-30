# 🍎 Fresh vs Rotten Fruit Image Classifier

A **Deep Learning image classification project** using a **Convolutional Neural Network (CNN)** to classify fruit images as **Fresh** or **Rotten**.

## 📌 Project Overview

This project uses **TensorFlow/Keras** to build and train a CNN-based binary image classifier.

The model processes fruit images and predicts:

* 🟢 **Fresh**
* 🔴 **Rotten**

The input images are resized to **224 × 224 pixels** and normalized to a **0–1** pixel range before prediction.

## 🧠 Technologies Used

* Python
* TensorFlow / Keras
* Convolutional Neural Network (CNN)
* NumPy
* Pillow
* Streamlit
* Scikit-learn
* Matplotlib

## ⚙️ Model Details

| Parameter     | Details                            |
| ------------- | ---------------------------------- |
| Model         | Convolutional Neural Network (CNN) |
| Task          | Binary Image Classification        |
| Input Size    | 224 × 224                          |
| Output        | Fresh / Rotten                     |
| Activation    | Sigmoid                            |
| Test Accuracy | **95.40%**                         |

## 🚀 Streamlit Application

The project includes an interactive **Streamlit web application** that allows users to:

1. Upload a fruit image
2. Preview the image
3. Run the CNN model
4. Get a Fresh/Rotten prediction
5. View the prediction confidence

## 📂 Project Structure

```text
Fresh-Rotten-Classifier/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

> The trained `.keras` model is not included directly in the GitHub repository because of GitHub file-size limitations. The model can be hosted separately and loaded by the Streamlit application.

## 💻 Run Locally

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
cd Fresh-Rotten-Classifier
```

### 2. Create a virtual environment

```bash
py -3.11 -m venv venv
```

### 3. Activate the environment

**Windows PowerShell:**

```bash
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

## 📊 Results

The trained CNN achieved **95.40% test accuracy** on the fruit classification task.

## 🎯 Key Skills Demonstrated

* Deep Learning
* CNN Architecture
* Image Classification
* Image Preprocessing
* TensorFlow/Keras
* Model Evaluation
* Python
* Streamlit
* Model Deployment

## 👩‍💻 Author

**Gargi Kundu**

B.Tech – Electronics & Communication Engineering (VLSI Design)

## 🔗 Links

* **GitHub:** https://github.com/Gargik283/fruit_classification_model_CNN
* **Live Demo:** https://fruitclassificationmodelcnn-l4pvfvrtjmcndvfs8fbvpa.streamlit.app/
