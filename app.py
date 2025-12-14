import time
import faiss
import pathlib
from PIL import Image
import numpy as np
import pandas as pd
import os

import streamlit as st
from streamlit_cropper import st_cropper

from features import EXTRACTORS

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
st.set_page_config(layout="wide")

FILES_PATH = str(pathlib.Path().resolve())

# Rutas a imágenes, base de datos y archivo CSV
IMAGES_PATH = os.path.join(FILES_PATH, "images")
DB_PATH = os.path.join(FILES_PATH, "database")
DB_FILE = "db.csv"


# Etiquetas
EXTRACTOR_LABELS = {
    "rgb_hist": "RGB Histogram (Color)",
    "hsv_hist": "HSV Histogram (Color)",
    "resnet18": "ResNet18 (CNN)",
    "efficientnet_b0": "EfficientNet-B0 (CNN)",
    "vgg19": "VGG19 (CNN)",
    "sift": "SIFT (Keypoints)",
}

# Orden desplegable (priorizar CNN frente a métodos clásicos)
PREFERRED_ORDER = [
    "resnet18",
    "efficientnet_b0",
    "vgg19",
    "hsv_hist",
    "rgb_hist",
    "sift",
]


def get_image_list():
    df = pd.read_csv(os.path.join(DB_PATH, DB_FILE))
    return list(df.image.values)


def build_options():
    available = set(EXTRACTORS.keys())

    ordered = [k for k in PREFERRED_ORDER if k in available]
    for k in sorted(available):
        if k not in ordered:
            ordered.append(k)

    labels = [EXTRACTOR_LABELS.get(k, k) for k in ordered]
    label_to_key = {EXTRACTOR_LABELS.get(k, k): k for k in ordered}
    return labels, label_to_key


def retrieve_image(img_query, extractor_key, n_imgs=11):
    extractor_fn = EXTRACTORS[extractor_key]
    index_path = os.path.join(DB_PATH, f"feat_{extractor_key}.index")

    # Comprueba que el índice existe
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"No encuentro el índice FAISS: {index_path}\n"
            f"Ejecuta: python build_db_and_index.py"
        )

    # Carga del índice FAISS
    indexer = faiss.read_index(index_path)

    if not isinstance(img_query, Image.Image):
        img_query = Image.fromarray(np.array(img_query))

    # Extracción de características
    vector = extractor_fn(img_query).astype("float32")
    # Normalización L2 para usar similitud coseno
    faiss.normalize_L2(vector)

    # Búsqueda
    _, indices = indexer.search(vector, k=n_imgs)
    return indices[0]


def main():
    st.title("CBIR IMAGE SEARCH")

    labels, label_to_key = build_options()

    col1, col2 = st.columns(2)

    with col1:
        st.header("QUERY")

        st.subheader("Choose feature extractor")
        option_label = st.selectbox(".", labels)

        extractor_key = label_to_key[option_label]
        st.caption(f"Using: `{extractor_key}`  →  index: `feat_{extractor_key}.index`")

        st.subheader("Upload image")
        img_file = st.file_uploader(label=".", type=["png", "jpg", "jpeg"])

        cropped_img = None
        if img_file:
            img = Image.open(img_file).convert("RGB")
            cropped_img = st_cropper(img, realtime_update=True, box_color="#FF0004")

            st.write("Preview")
            preview_img = cropped_img.copy()
            preview_img.thumbnail((150, 150))
            st.image(preview_img)

    with col2:
        st.header("RESULT")
        if img_file and cropped_img is not None:
            st.markdown("**Retrieving .......**")
            start = time.time()

            retriev = retrieve_image(cropped_img, extractor_key, n_imgs=11)
            image_list = get_image_list()

            end = time.time()
            st.markdown("**Finish in " + str(end - start) + " seconds**")

            col3, col4 = st.columns(2)

            with col3:
                image = Image.open(os.path.join(IMAGES_PATH, image_list[retriev[0]])).convert("RGB")
                st.image(image, use_container_width=True)

            with col4:
                image = Image.open(os.path.join(IMAGES_PATH, image_list[retriev[1]])).convert("RGB")
                st.image(image, use_container_width=True)

            col5, col6, col7 = st.columns(3)

            with col5:
                for u in range(2, 11, 3):
                    image = Image.open(os.path.join(IMAGES_PATH, image_list[retriev[u]])).convert("RGB")
                    st.image(image, use_container_width=True)

            with col6:
                for u in range(3, 11, 3):
                    image = Image.open(os.path.join(IMAGES_PATH, image_list[retriev[u]])).convert("RGB")
                    st.image(image, use_container_width=True)

            with col7:
                for u in range(4, 11, 3):
                    image = Image.open(os.path.join(IMAGES_PATH, image_list[retriev[u]])).convert("RGB")
                    st.image(image, use_container_width=True)


if __name__ == "__main__":
    main()
