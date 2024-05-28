import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import streamlit as st  
import tensorflow as tf 
from PIL import Image
import numpy as np



import warnings
warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="Reconocimiento de objetos",
    page_icon = ":smile:",
    initial_sidebar_state = 'auto'
)

hide_streamlit_style = """
	<style>
  #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
  </style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

@st.cache_resource
def load_model():
    model=tf.keras.models.load_model('./objetos.h5')
    return model
with st.spinner('Modelo está cargando..'):
    model=load_model()
    


with st.sidebar:
        st.image('shampo.png')
        st.title("Reconocimiento de imagen")
        st.subheader("Reconocimiento de imagen para objetos")

st.image('logoA.png')
st.title("Proyecto Ilona Pava")
st.write("Proyecto desarrollado para predecir a partir de una foto,una prediccion que tiene como posible resultados 10 objetos de aseo personal")
st.write("""
         # Detección de objetos
         """
         )


def import_and_predict(image_data, model, class_names):
    
    image_data = image_data.resize((180, 180))
    
    image = tf.keras.utils.img_to_array(image_data)
    image = tf.expand_dims(image, 0) # Create a batch

    
    # Predecir con el modelo
    prediction = model.predict(image)
    index = np.argmax(prediction)
    score = tf.nn.softmax(prediction[0])
    class_name = class_names[index]
    
    return class_name, score


class_names = open("./clases.txt", "r").readlines()

img_file_buffer = st.camera_input("Capture una foto para identificar una imagen")    
if img_file_buffer is None:
    st.text("Por favor tome una foto")
else:
    image = Image.open(img_file_buffer)
    st.image(image, use_column_width=True)
    
    # Realizar la predicción
    class_name, score = import_and_predict(image, model, class_names)
    
    # Mostrar el resultado

    if np.max(score)>0.5:
        st.subheader(f"Tipo de objeto: {class_name}")
        st.text(f"Con una puntuación de confianza: {100 * np.max(score):.2f}%")
    else:
        st.text(f"No se pudo determinar el tipo de flor")