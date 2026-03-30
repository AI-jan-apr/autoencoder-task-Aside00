# 🔍 Digit Similarity Finder (Autoencoder Project)

An intelligent deep learning project that uses an **Autoencoder Neural Network** to find visually similar handwritten digits from the MNIST dataset.

---

## 🚀 Project Overview

This project demonstrates how a trained autoencoder can learn compact representations (latent space) of images and use them for **image similarity search**.

Instead of classifying digits, the model learns to **understand visual structure** and retrieve similar handwritten digits.

---

## 🧠 How It Works

1. A trained **encoder model** compresses images into a lower-dimensional representation.
2. Uploaded images are processed and encoded.
3. The system compares the encoded vector with MNIST dataset encodings.
4. It retrieves the **Top 5 most similar digits** using cosine similarity.

---

## 🎯 Key Features

- 📤 Upload handwritten digit images
- 🔍 Find visually similar digits from MNIST dataset
- 🧠 Deep Autoencoder-based feature extraction
- 📊 t-SNE visualization (optional in notebook version)
- ⚡ Fast similarity search using cosine similarity
- 🎨 Clean Streamlit web interface

---

## 🧪 My Experiment

As part of testing the model:

- I uploaded a **printed digit "5"**
- The model successfully processed it and understood its structure
- It then retrieved and matched it with **handwritten "5" digits**
- The system correctly identified and returned similar handwritten versions of the digit

<img width="1900" height="957" alt="image" src="https://github.com/user-attachments/assets/592f2d02-2683-44a2-aedb-eb35fbb0b96f" />


👉 This shows that the model does not rely on exact pixel matching, but instead understands the **underlying shape and pattern of digits**.

---

## 🛠️ Tech Stack

- Python 🐍
- TensorFlow / Keras
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- MNIST Dataset

---

## 📁 Project Structure
autoencoder-task/
│
├── app.py # Streamlit web app
├── model/
│ ├── encoder.h5 # Trained encoder model
│ └── autoencoder.h5 # Full autoencoder model
│
└── README.md



---

## ▶️ How to Run

```bash
pip install streamlit tensorflow numpy scikit-learn pillow
streamlit run app.py
