# Waste Classifier · Clasificación de residuos

Image classifier that tells you which Peruvian waste bin (NTP 900.058:2019) an item goes in. An EfficientNetB3 network fine-tuned with transfer learning at 300 px recognizes 8 waste classes and a Streamlit app shows the verdict with its confidence. A lighter MobileNetV2 model is included as an alternative.

*Clasificador de imágenes que indica en qué tacho (NTP 900.058:2019) va un residuo. Una red EfficientNetB3 con transfer learning a 300 px reconoce 8 clases y una app en Streamlit muestra el resultado con su confianza. Se incluye también un modelo MobileNetV2 más liviano.*

<!-- Add a screenshot of the deployed app here: ![app](docs/app.png) -->
**Live app:** _add your Streamlit link here_

---

## English

### Results
Both improved models were trained on 8 720 images and evaluated once on the same independent test set of 1 869 images (stratified 70/15/15 split, seed 42, exact duplicates removed).

| Model | Test accuracy | Macro F1 | Weakest class F1 | Inference (Colab GPU) |
|---|---|---|---|---|
| **EfficientNetB3, 300 px + CutMix (default)** | **0.9647** (0.9652 with TTA) | **0.965** | 0.940 (`trash`) | 88 ms |
| MobileNetV2, 224 px (light option) | 0.9294 (0.9299 with TTA) | 0.928 | 0.885 (`trash`) | 79 ms |

- EfficientNetB3 training: 33 epochs (8 head + 25 fine-tuning), 48.8 min on a Colab T4. The reloaded model reproduces the validation accuracy of training (0.9700).
- MobileNetV2 5-fold cross-validation on frozen features: 0.9255 ± 0.0048.
- An ensemble of both models (weight chosen on validation) reached 0.9642 and did not beat EfficientNetB3 alone, so it is not used.
- Earlier baselines on a different 80/20 split (not strictly comparable): HOG + LinearSVC 0.4270, MobileNetV2 (first run) 0.9168, EfficientNetB3 at 224 px 0.9032.
- Each model is a single run. The changes (300 px, CutMix, deeper fine-tuning) were applied together, so the gain of each one was not measured separately. Some of the remaining errors look like ambiguous or doubtful dataset labels.

### What is inside
- **Classes:** battery, biological, cardboard, glass, metal, paper, plastic, trash → mapped to the 4 bins of the Peruvian standard (green, black, brown, red).
- **Training pipeline:** `tf.data` with cache, in-model preprocessing, mixed precision, class weights, label smoothing, CutMix, two-phase fine-tuning with frozen BatchNorm, model selection by validation and test-time augmentation.
- **Dataset:** Garbage Classification V2 (Kaggle), filtered to 8 classes (12 458 images). Check the dataset license on Kaggle before redistributing it; the images are not included in this repository.

### Run it
```bash
pip install -r requirements.txt
streamlit run app.py
```
The trained models are in `models/`. The app reads the input size from the model, so it works with both. To evaluate on a folder of test images:
`python src/evaluar_modelo.py path/to/test models/efficientnetb3_mejorado.keras interno`

### Models
| File | Size | Origin |
|---|---|---|
| `models/efficientnetb3_mejorado.keras` (default) | 46 MB | `notebooks/EfficientNetB3_300_CutMix_Ensamble.ipynb` (Colab, GPU), re-saved with `include_optimizer=False` |
| `models/mobilenetv2_mejorado.keras` | 36 MB | `notebooks/MobileNetV2_Entrenamiento_Mejorado.ipynb` |

GitHub rejects files over 100 MB; saving without the optimizer state reduced the EfficientNetB3 file from about 179 MB to 46 MB.

### Structure
`app.py` · `src/` (evaluation and training scripts) · `notebooks/` (Colab: EfficientNetB3 300 px + CutMix, improved MobileNetV2, original MobileNetV2 + baseline, EfficientNetB3 first run) · `models/` · `tests_imgs/`

### Limitations
Public dataset: photos taken with a phone in the real world may perform worse. The tool supports a decision and can be wrong, especially with items outside the 8 classes.

---

## Español

### Resultados
Los dos modelos mejorados se entrenaron con 8 720 imágenes y se evaluaron una sola vez en el mismo conjunto de prueba independiente de 1 869 imágenes (partición estratificada 70/15/15, semilla 42, sin duplicados exactos).

| Modelo | Exactitud en prueba | Macro F1 | F1 clase más débil | Tiempo por imagen (GPU Colab) |
|---|---|---|---|---|
| **EfficientNetB3, 300 px + CutMix (por defecto)** | **0,9647** (0,9652 con TTA) | **0,965** | 0,940 (`trash`) | 88 ms |
| MobileNetV2, 224 px (opción ligera) | 0,9294 (0,9299 con TTA) | 0,928 | 0,885 (`trash`) | 79 ms |

- Entrenamiento de EfficientNetB3: 33 épocas (8 de cabeza y 25 de fine-tuning), 48,8 min en una T4 de Colab. El modelo recargado reproduce la exactitud de validación del entrenamiento (0,9700).
- Validación cruzada de 5 pliegues de MobileNetV2: 0,9255 ± 0,0048.
- El ensamble de ambos modelos obtuvo 0,9642 y no superó a EfficientNetB3 solo, por lo que no se usa.
- Referencias anteriores con otra partición 80/20 (no estrictamente comparables): HOG + LinearSVC 0,4270, MobileNetV2 (primera corrida) 0,9168, EfficientNetB3 a 224 px 0,9032.
- Cada modelo es una sola corrida. Los cambios (300 px, CutMix, fine-tuning más profundo) se aplicaron juntos, así que no se midió el aporte de cada uno. Parte de los errores restantes parecen imágenes ambiguas o etiquetas dudosas del dataset.

### Cómo ejecutarlo
```bash
pip install -r requirements.txt
streamlit run app.py
```
Los modelos ya están en `models/`. La app toma el tamaño de entrada del propio modelo.

---

Author: **Fernando Olazabal** · License: MIT
