"""
Script para construir la base de datos y los índices FAISS.
Genera el archivo db.csv a partir de la carpeta images/ y crea un índice
FAISS independiente para cada extractor de características.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import faiss

from features import EXTRACTORS

IMAGES_PATH = Path("images")

# Carpeta donde guardamos la base de datos y los índices FAISS
DB_PATH = Path("database")
DB_PATH.mkdir(exist_ok=True)

def build_db():
    """
    Recorre la carpeta images/clase/archivo y construye un CSV (db.csv) con dos columnas:
      - image: ruta relativa "clase/archivo"
      - label: nombre de la clase (carpeta)
    """
    rows = []
    for class_name in sorted(os.listdir(IMAGES_PATH)):
        class_dir = IMAGES_PATH / class_name
        if not class_dir.is_dir():
            continue                # Si no es una carpeta, la ignoramos

        # Recorremos los ficheros dentro de cada clase
        for f in os.listdir(class_dir):
            if f.lower().endswith((".jpg",".jpeg",".png")):
                rows.append({"image": f"{class_name}/{f}", "label": class_name})
    
    # Guardamos como CSV
    df = pd.DataFrame(rows)
    df.to_csv(DB_PATH / "db.csv", index=False)
    return df

def build_index(df, name, extractor_fn):
    """
    Construye un índice FAISS para cada extractor.
    Pasos:
      1) Recorrer todas las imágenes de la galería (df)
      2) Extraer su vector de características con extractor_fn
      3) Apilar todos los vectores en una matriz (N, D)
      4) Normalizar L2 (para poder usar similitud coseno)
      5) Crear un índice exacto IndexFlatIP y añadir los vectores
      6) Guardar el índice en database/feat_<name>.index
    """
    feats = []

    # Extraemos características de cada imagen
    for i, row in df.iterrows():
        img = Image.open(IMAGES_PATH / row["image"]).convert("RGB")
        v = extractor_fn(img)      # extractor_fn devuelve (1, D), por eso cogemos v[0]->(D,)
        feats.append(v[0])
    
    # Convertimos la lista de vectores a una matriz (N, D)
    feats = np.vstack(feats).astype("float32")

    # Similitud coseno en FAISS:: normalizamos + IndexFlatIP
    faiss.normalize_L2(feats)
    d = feats.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(feats)

    faiss.write_index(index, str(DB_PATH / f"feat_{name}.index"))
    print(f"[OK] index {name}: N={feats.shape[0]} D={d}")

def main():
    df = build_db()
    print("[OK] db.csv:", len(df), "imágenes")

    for name, fn in EXTRACTORS.items():
        build_index(df, name, fn)

if __name__ == "__main__":
    main()
