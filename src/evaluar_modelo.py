"""Evalúa un modelo sobre una carpeta de imágenes organizada por clase (p. ej. el conjunto de prueba).
Uso:  python src/evaluar_modelo.py ruta/test models/efficientnetb3_residuos.h5 efficientnet
      python src/evaluar_modelo.py ruta/test models/mobilenetv2_residuos.h5 mobilenet
      python src/evaluar_modelo.py ruta/test models/mobilenetv2_mejorado.keras interno   # preprocesa dentro del modelo
"""
import sys, json, numpy as np, tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

ruta, modelo_path, tipo = sys.argv[1], sys.argv[2], sys.argv[3]
clases = json.load(open("clases.json", encoding="utf-8"))["class_names"]
modelo = tf.keras.models.load_model(modelo_path, compile=False)
ds = tf.keras.utils.image_dataset_from_directory(
    ruta, labels="inferred", label_mode="int", class_names=clases,
    image_size=(224, 224), batch_size=32, shuffle=False)
y_true = np.concatenate([y.numpy() for _, y in ds])
escala = (lambda x: x) if tipo in ("efficientnet", "interno") else (lambda x: x / 255.0)   # igual que en el entrenamiento
y_pred = np.argmax(modelo.predict(ds.map(lambda x, y: escala(x)), verbose=0), axis=1)
print(classification_report(y_true, y_pred, target_names=clases, digits=3))
print(confusion_matrix(y_true, y_pred))
