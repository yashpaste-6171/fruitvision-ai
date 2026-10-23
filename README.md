# 🍎 FruitVision AI — Fruit Classification Using Deep Learning

## 📌 Project Overview

FruitVision AI is a computer vision project that classifies fruit images using a deep learning model based on **EfficientNetB0** and transfer learning.

Users can upload a fruit image through a Streamlit web application and receive a predicted fruit class with confidence scores.

## 🎯 Objectives

- Acquire and preprocess a fruit image dataset.
- Explore classical computer vision techniques.
- Train a deep learning model for fruit classification.
- Evaluate model performance.
- Deploy an interactive fruit classification web application.

## 📂 Dataset

**Dataset:** [Fruit Recognition — Kaggle](https://www.kaggle.com/datasets/sshikamaru/fruit-recognition)

The dataset contains 33 fruit classes in the labeled training collection.

The project includes:
- Image resizing and normalization demonstrations
- Corrupted-image and duplicate-image checks
- Dataset splitting into training, validation, and testing subsets
- Data augmentation demonstrations

## 🧠 Model

- **Architecture:** EfficientNetB0
- **Approach:** Transfer learning with pretrained ImageNet weights
- **Input size:** 224 × 224 pixels
- **Output:** 33 fruit classes
- **Framework:** TensorFlow / Keras

## 🔬 Computer Vision Techniques

The project explores three classical computer vision techniques:

1. **Harris Corner Detection** — detects corner-like image structures.
2. **ORB Feature Extraction** — detects keypoints and computes binary descriptors.
3. **Canny Edge Detection** — identifies image boundaries and edges.

## 🌐 Web Application

The interactive application is built with Streamlit.

**Live Demo:** https://fruitvision-ai-4iu6bnr5xnvgfrpw5qloct.streamlit.app

### How to use

1. Open the live application.
2. Upload a fruit image in JPG, JPEG, or PNG format.
3. Click **Classify Fruit**.
4. View the predicted fruit class and confidence scores.

## 🛠️ Technologies Used

- Python
- TensorFlow and Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Streamlit
- Google Colab
- GitHub

## 🚀 Run the Application Locally

Clone the repository:

```bash
git clone https://github.com/yashpaste-6171/fruitvision-ai.git
cd fruitvision-ai
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit:

```bash
streamlit run app.py
```

The application will display a local URL in your terminal.

## 📊 Evaluation

The model was evaluated using a held-out labeled test split. Evaluation outputs include accuracy and loss, accuracy and loss graphs, a confusion matrix, and a classification report.

Predictions on new real-world fruit photographs may differ from results on the dataset.

## 📁 Repository Contents

- `app.py` — Streamlit application
- `fruit_classifier_final.keras` — trained classification model
- `fruit_class_names.json` — class labels used by the application
- `requirements.txt` — Python dependencies

## 👨‍💻 Author

**Yash Paste**

Computer Vision Midterm Project

## ⚠️ Disclaimer

This project is developed for educational purposes as part of a computer vision assignment. Predictions are model estimates and may not always be correct.
