import os

# Configurar TensorFlow para reducir mensajes y desactivar oneDNN si es necesario
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Oculta mensajes de nivel informativo y advertencias
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Desactiva las optimizaciones de oneDNN

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # Corrige la advertencia con tf.compat.v1

from deepface import DeepFace

# Análisis de la imagen con acciones especificadas
try:
    result = DeepFace.analyze(
        img_path="12345.jpg",
        actions=['emotion',  'age', 'gender', 'race']  # Prueba con estas acciones inicialmente
    )
    print("Resultados del análisis:", result)
except Exception as e:
    print("Error al realizar el análisis:", e)
