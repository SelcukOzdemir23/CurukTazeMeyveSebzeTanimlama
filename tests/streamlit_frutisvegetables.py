import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
#from tensorflow.keras.applications.resnet50 import preprocess_input
import plotly.express as px
import os

# model yükle
model = tf.keras.models.load_model("tests\\fine_tuned_model.h5")

# Etiketler
waste_labels = {0: 'taze_elma',
                 1: 'taze_muz',
                   2: 'taze_acı_kabak',
                     3: 'taze_kırmızıbiber',
                     4: 'taze_portakal',
                     5: 'taze_domates',
                     6: 'bayat_elma',
                     7: 'bayat_muz',
                     8: 'stale_bitter_gourd',
                     9: 'bayat_acı_kabak',
                     10: 'bayat_portakal',
                     11: 'bayat_domates',
                     
                     
                     
                     }



# uygulama yükle
st.title("Çürük - Taze Meyve ve Sebze Tahmin Uygulaması")
st.write("Lütfen bir meyve ya da sebze yükleyin")

# giriş yap
uploaded_image = st.file_uploader("Meyve ya da sebze yükle! ", type=["jpg", "png", "jpeg"])

# resim işleme
# def file_selector(folder_path='.'):
#     filenames = os.listdir(folder_path)
#     selected_filename = st.selectbox('Select a file', filenames)
#     return os.path.join(folder_path, selected_filename)

# filename = file_selector()
# st.write('You selected `%s`' % filename)

if uploaded_image is not None:
    # Görüntüyü modelin girdi boyutuna yeniden boyutlandırın
    img = image.load_img(uploaded_image, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    

    # tahmin
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    # Sonuç
    st.image(uploaded_image, caption='Yüklenen Görüntü', use_column_width=True)
    st.write(f"Tahmin Edilen Sınıf: {waste_labels[predicted_class]}")

    # görselleştirme
    st.write("Tahmin İhtimalleri:")
    labels = list(waste_labels.values())
    probabilities = prediction[0] * 100  # İhtimalleri yüzde olarak hesapla

    # Çubuk grafik
    fig_bar = px.bar(x=labels, y=probabilities, labels={'x': 'Sınıf', 'y': 'Yüzde (%)'},
                     title="Tahmin İhtimalleri (Çubuk Grafik)")
    st.plotly_chart(fig_bar)

    # Pasta grafiği
    fig_pie = px.pie(values=probabilities, names=labels, title="Tahmin İhtimalleri (Pasta Grafiği)")
    st.plotly_chart(fig_pie)

