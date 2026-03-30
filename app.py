import streamlit as st
import numpy as np
import os
from tensorflow.keras.models import load_model
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity


model_path = os.path.join("model", "encoder.h5")
encoder = load_model(model_path)


from tensorflow.keras.datasets import mnist
(XTrain, _), _ = mnist.load_data()

XTrain = XTrain.astype('float32') / 255
XTrain_flat = XTrain.reshape((len(XTrain), 784))

encoded_train = encoder.predict(XTrain_flat)


st.set_page_config(page_title="Digit Similarity Finder", layout="wide")

st.title("🔢 Digit Similarity Finder")
st.write("ارفع صورة رقم (handwritten digit) وراح نجيب لك أقرب 5 أرقام مشابهة")

uploaded_file = st.file_uploader("📤 ارفع صورة رقم", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    image = image.resize((28, 28))

    img_array = np.array(image)

    img_array = 255 - img_array

    
    img_array = img_array.astype('float32') / 255
    img_flat = img_array.reshape(1, 784)

    query_encoded = encoder.predict(img_flat)

    similarities = cosine_similarity(query_encoded, encoded_train)
    top5 = np.argsort(similarities[0])[-5:]

   
    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("📌 الصورة المدخلة")
        st.image(img_array, width=150)

    with col2:
        st.subheader("🔍 أقرب 5 صور مشابهة")
        cols = st.columns(5)

        for i, idx in enumerate(top5):
            with cols[i]:
                st.image(XTrain[idx], use_column_width=True)