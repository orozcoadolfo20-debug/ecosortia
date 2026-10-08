import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import keras # <--- 1. AGREGAR ESTA LÍNEA
import os

# 1. Configuración general de la página (Debe ir siempre de primero)
st.set_page_config(
    page_title="EcoSort IA - Clasificador Inteligente",
    page_icon="logo.jpg", 
    layout="centered"
)

# 2. PARCHE PARA MODELOS DE TEACHABLE MACHINE EN TENSORFLOW NUEVO
# <--- 2. USAR keras DIRECTAMENTE SIN EL PREFIJO tf.
class CustomDepthwiseConv2D(keras.layers.DepthwiseConv2D):
    def __init__(self, **kwargs):
        if 'groups' in kwargs:
            del kwargs['groups']
        super().__init__(**kwargs)

# 3. Inicializar la memoria (Contadores)
if 'total_residuos' not in st.session_state:
    st.session_state.total_residuos = 0
if 'conteo_categorias' not in st.session_state:
    st.session_state.conteo_categorias = {}

# 4. Rutas de los archivos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'keras_model.h5')
LABELS_PATH = os.path.join(BASE_DIR, 'labels.txt')
LOGO_PATH = os.path.join(BASE_DIR, 'logo.jpg') # <-- Modificado para JPG

# 5. Funciones para cargar el modelo y las etiquetas
@st.cache_resource
def load_model():
    # Inyectamos el parche al cargar el modelo
    model = tf.keras.models.load_model(
        MODEL_PATH, 
        compile=False,
        custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D}
    )
    return model

def load_labels():
    with open(LABELS_PATH, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

# 6. Panel lateral (Sidebar) para mostrar las estadísticas
try:
    st.sidebar.image(LOGO_PATH, width="stretch")
except FileNotFoundError:
    st.sidebar.warning("⚠️ Falta el archivo logo.jpg") # <-- Modificado para JPG

st.sidebar.title("📊 Panel de Estadísticas")
st.sidebar.metric(label="Total Clasificados", value=st.session_state.total_residuos)

st.sidebar.markdown("---")
st.sidebar.markdown("**Desglose por material:**")
if st.session_state.total_residuos == 0:
    st.sidebar.info("Aún no se han registrado residuos.")
else:
    for categoria, cantidad in st.session_state.conteo_categorias.items():
        st.sidebar.write(f"- **{categoria}**: {cantidad}")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reiniciar Contadores"):
    st.session_state.total_residuos = 0
    st.session_state.conteo_categorias = {}
    st.rerun()

# 7. Encabezado principal
col1, col2 = st.columns([1, 3])
with col1:
    try:
        st.image(LOGO_PATH, width=180)
    except:
        pass
with col2:
    st.title("EcoSort IA: Clasificación con Inteligencia Artificial")
    st.write("Identifica el tipo de residuo y descubre cómo reciclarlo correctamente.")

# 8. Carga del modelo
try:
    model = load_model()
    class_names = load_labels()
except Exception as e:
    st.warning(f"⚠️ Error al cargar: {e}. Revisa que keras_model.h5 y labels.txt estén en la carpeta.")
    model = None

# 9. Función central para procesar la imagen
def classify_image(img):
    size = (224, 224)
    image_resized = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image_resized)
    
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.expand_dims(normalized_image_array, axis=0)
    
    prediction = model.predict(data)
    index = np.argmax(prediction)
    confidence_score = prediction[0][index]
    
    return class_names[index], confidence_score

# Función para registrar en la memoria
def registrar_residuo(categoria):
    st.session_state.total_residuos += 1
    if categoria in st.session_state.conteo_categorias:
        st.session_state.conteo_categorias[categoria] += 1
    else:
        st.session_state.conteo_categorias[categoria] = 1

# 10. Interfaz de Usuario con Pestañas
tab1, tab2 = st.tabs(["📷 Cámara en Vivo", "📂 Subir Imagen"])

# --- Pestaña 1: Cámara en Vivo ---
with tab1:
    st.write("Utiliza la cámara para analizar el desecho al instante.")
    camera_image = st.camera_input("Toma una foto del residuo")
    
    if camera_image is not None and model is not None:
        image = Image.open(camera_image)
        st.write("🔄 Procesando imagen...")
        
        predicted_class, confidence = classify_image(image)
        label_text = predicted_class.split(" ", 1)[-1] if " " in predicted_class else predicted_class
        
        st.success("¡Análisis completado!")
        st.markdown(f"### 🎯 Categoría Detectada: **{label_text}** (`{confidence * 100:.2f}%` certeza)")
        
        # Botón para confirmar y sumar al contador
        if st.button("➕ Registrar en Estadísticas", key="btn_cam"):
            registrar_residuo(label_text)
            st.rerun()

# --- Pestaña 2: Subir un Archivo ---
with tab2:
    st.write("Sube una fotografía de tu galería.")
    uploaded_file = st.file_uploader("Selecciona una imagen...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None and model is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen a analizar", width="stretch")
        st.write("🔄 Procesando imagen...")
        
        predicted_class, confidence = classify_image(image)
        label_text = predicted_class.split(" ", 1)[-1] if " " in predicted_class else predicted_class
        
        st.success("¡Análisis completado!")
        st.markdown(f"### 🎯 Categoría Detectada: **{label_text}** (`{confidence * 100:.2f}%` certeza)")
        
        # Botón para confirmar y sumar al contador
        if st.button("➕ Registrar en Estadísticas", key="btn_file"):
            registrar_residuo(label_text)
            st.rerun()

# Pie de página
st.markdown("---")
st.caption("Proyecto Feria Tecnológica Industria 4.0: Humanos e IA trabajando juntos.")
