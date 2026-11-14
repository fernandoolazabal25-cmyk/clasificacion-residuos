# Waste Classifier · Clasificación de residuos

Image classifier that tells you which Peruvian waste bin (NTP 900.058:2019) an item goes in. A MobileNetV2 network fine-tuned with transfer learning recognizes 8 waste classes and a Streamlit app shows the verdict with its confidence.

*Clasificador de imágenes que indica en qué tacho (NTP 900.058:2019) va un residuo. Una red MobileNetV2 con transfer learning reconoce 8 clases y una app en Streamlit muestra el resultado con su confianza.*

<!-- Add a screenshot of the deployed app here: ![app](docs/app.png) -->
**Live app:** _add your Streamlit link here_

---

## English

### Results
Final model: MobileNetV2 with fine-tuning, trained on 8 720 images and evaluated once on an independent test set of 1 869 images (stratified 70/15/15 split, exact duplicates removed).

| Metric | Value |
|---|---|
| Test accuracy | 0.9294 (0.9299 with test-time augmentation) |
| Test macro F1 | 0.9282 |
| Weakest class F1 (`trash`) | 0.885 |
| 5-fold cross-validation (on frozen features) | 0.9255 ± 0.0048 |
| Inference time | 79 ms per image (Colab GPU) |
| Training time | 7.8 min (Colab GPU) |

Comparison on a different 80/20 split (not strictly comparable): HOG + LinearSVC baseline 0.4270, EfficientNetB3 0.9032, MobileNetV2 (original run) 0.9168. Each model was trained once, so differences of one or two points are not conclusive.

### What is inside
- **Classes:** battery, biological, cardboard, glass, metal, paper, plastic, trash → mapped to the 4 bins of the Peruvian standard (green, black, brown, red).
- **Training pipeline:** `tf.data` with cache, in-model augmentation and preprocessing, mixed precision, class weights, label smoothing, two-phase fine-tuning with frozen BatchNorm, model selection by validation, cross-validation and test-time augmentation.
- **Dataset:** Garbage Classification V2 (Kaggle), filtered to 8 classes (12 458 images). Check the dataset license on Kaggle before redistributing it; the images are not included in this repository.

### Run it
```bash
pip install -r requirements.txt
streamlit run app.py
```
Put the trained model in `models/` first (see below). To evaluate on a folder of test images: `python src/evaluar_modelo.py path/to/test models/mobilenetv2_mejorado.keras interno`

### Models
Not stored in git history by default. Place the files in `models/`:

| File | How to get it |
|---|---|
| `mobilenetv2_mejorado.keras` (default) | Output of `notebooks/MobileNetV2_Entrenamiento_Mejorado.ipynb` (Colab, GPU) |
| `mobilenetv2_residuos.h5` (optional) | First MobileNetV2 run |
| `efficientnetb3_residuos.h5` (optional) | EfficientNetB3 comparison run |

GitHub rejects files over 100 MB; save with `model.save(path, include_optimizer=False)`.

### Structure
`app.py` · `src/` (evaluation and training scripts) · `notebooks/` (Colab: improved training, original MobileNetV2 + baseline, EfficientNetB3) · `models/` · `tests_imgs/`

---

## Español

### Resultados
Modelo final: MobileNetV2 con fine-tuning, entrenado con 8 720 imágenes y evaluado una sola vez en un conjunto de prueba independiente de 1 869 imágenes (partición estratificada 70/15/15, sin duplicados exactos).

| Métrica | Valor |
|---|---|
| Exactitud en prueba | 0,9294 (0,9299 con TTA) |
| Macro F1 en prueba | 0,9282 |
| F1 de la clase más débil (`trash`) | 0,885 |
| Validación cruzada de 5 pliegues | 0,9255 ± 0,0048 |
| Tiempo por imagen | 79 ms (GPU de Colab) |
| Tiempo de entrenamiento | 7,8 min (GPU de Colab) |

La comparación con el baseline (HOG + LinearSVC, 0,4270), EfficientNetB3 (0,9032) y la primera corrida de MobileNetV2 (0,9168) usa otra partición 80/20, así que no es estrictamente comparable y las diferencias de uno o dos puntos no son concluyentes.

### Cómo ejecutarlo
```bash
pip install -r requirements.txt
streamlit run app.py
```
Antes copia el modelo entrenado a `models/` (ver la tabla de modelos arriba).

---

Author: **Fernando Olazabal** · License: MIT
