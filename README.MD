# CBIR – Content-Based Image Retrieval System

## Integrantes del grupo P

- **Natalia Figal Izquierdo**
- **Gloria Domingo de Loresecha**

---

## Descripción del proyecto

Este repositorio contiene la implementación completa de un sistema de **Recuperación de Imágenes Basado en el Contenido (CBIR)** aplicado a un conjunto de imágenes de frutas.  
El sistema permite comparar distintos **extractores de características visuales**, tanto clásicos como basados en **redes neuronales convolucionales (CNN)**, y realizar búsquedas eficientes mediante **FAISS**.

El flujo completo del sistema incluye:

- Preparación y particionado del dataset.
- Extracción de características visuales.
- Indexación mediante FAISS.
- Evaluación cuantitativa y cualitativa.
- Interfaz gráfica interactiva desarrollada con **Streamlit**.

---

## Dataset

El dataset está compuesto por imágenes de cinco clases de frutas:

- Apple
- Banana
- Grape
- Mango
- Strawberry

El conjunto se divide en:

- **TRAIN (galería)**: imágenes utilizadas para construir los índices FAISS.
- **TEST**: imágenes independientes utilizadas para evaluar el sistema.

> **Nota importante**  
> Este repositorio **ya incluye**:
>
> - El archivo `database/db.csv`
> - Todos los índices FAISS (`database/feat_*.index`)
>
> Por tanto, **no es necesario reconstruir la base de datos ni los índices** para probar el sistema. Basta con ejecutar la aplicación.

---

## Extractores implementados

El sistema permite seleccionar dinámicamente el extractor de características entre los siguientes:

- EfficientNet-B0 (CNN)
- ResNet18 (CNN)
- VGG19 (CNN)
- SIFT (Keypoints)
- HSV Histogram (Color)
- RGB Histogram (Color)

Cada extractor cuenta con su índice FAISS independiente para garantizar comparaciones justas y reproducibles.

---

## Requisitos y entorno

Durante el desarrollo se intentó utilizar Python 3.12, pero se detectaron problemas de compatibilidad (principalmente errores de DLL) con algunas librerías de visión por computador y deep learning en Windows.

Para asegurar la estabilidad del sistema, se ha utilizado:

- Miniconda
- Python 3.10

Este entorno permite ejecutar correctamente PyTorch, torchvision, OpenCV y FAISS en Windows.

---

## Clonar el repositorio

Primero, clona el repositorio en tu máquina local:

```bash
git clone https://github.com/glodolo/Proyecto-CBIR/tree/main 
cd Proyecto-CBIR
```

--- 

## Instalación de dependencias

Una vez creado y activado el entorno con Python 3.10, instalar las dependencias con:

```bash
pip install -r requirements.txt

```

## Ejecución de la aplicación

Para lanzar la interfaz gráfica del sistema CBIR:

```bash
streamlit run app.py
```


Se abrirá una aplicación web local desde la que es posible:

- Seleccionar el extractor de características.
- Subir una imagen de consulta.
- Visualizar las imágenes más similares recuperadas.

--- 

## Estructura del proyecto

```TEXT
Proyecto-CBIR/
│
├── app.py                     # Interfaz gráfica (Streamlit)
├── build_db_and_index.py      # Construcción de la base de datos e índices FAISS
├── evaluate.py                # Evaluación cuantitativa (Prec@5, mAP@5)
├── features.py                # Definición de los extractores de características
├── requirements.txt           # Dependencias del proyecto
├── README.md                  # Este archivo
│
├── database/
│   ├── db.csv
│   ├── feat_efficientnet_b0.index
│   ├── feat_resnet18.index
│   ├── feat_vgg19.index
│   ├── feat_sift.index
│   ├── feat_hsv_hist.index
│   ├── feat_rgb_hist.index
│   └── metrics_train_test_prec_map_at5.csv
│
├── images/                    # Galería TRAIN (organizada por clases)
│   ├── Apple/
│   ├── Banana/
│   ├── Grape/
│   ├── Mango/
│   └── Strawberry/
│
└── images_test/               # Conjunto TEST (organizado por clases)
    ├── Apple/
    ├── Banana/
    ├── Grape/
    ├── Mango/
    └── Strawberry/

```
