"""
Este script evalúa el sistema CBIR de forma cuantitativa usando:
- Precision@K
- mAP@K (mean Average Precision@K)

Se calcula el rendimiento en:
1) TRAIN (usando como queries las propias imágenes de la galería). Aquí se evita el self-match
   pidiendo K+1 resultados y descartando el primero.
2) TEST (usando imágenes independientes de images_test/).

Salida:
- Imprime una tabla comparativa por extractor
- Guarda un CSV con las métricas en database/
"""

import os
from pathlib import Path
import pandas as pd
from PIL import Image
import faiss

from features import EXTRACTORS

IMAGES_PATH = Path("images")        # galería (train)
TEST_PATH = Path("images_test")     # queries de test
DB_PATH = Path("database")


def precision_at_k(retrieved_labels, true_label, k):

    """
    Precision@K: proporción de resultados relevantes dentro del top-k.
    Un resultado es relevante si su label coincide con la clase real de la query.
    """

    return sum(1 for lab in retrieved_labels[:k] if lab == true_label) / k


def ap_at_k(retrieved_labels, true_label, k):

    """
    Average Precision@K para una query.
    - Relevante si label == true_label
    - AP@K = media de las precisiones en las posiciones donde aparece un relevante
    (tiene en cuenta el orden dentro del ranking)
    """

    hits = 0
    sum_prec = 0.0

    # Recorremos las predicciones del top k y acumulamos precisión cuando acertamos
    for i, lab in enumerate(retrieved_labels[:k], start=1):
        if lab == true_label:
            hits += 1
            sum_prec += hits / i
    return sum_prec / hits if hits > 0 else 0.0


def load_gallery():

    """
    Carga database/db.csv, que contiene la lista de imágenes indexadas y su etiqueta.
    Devuelve:
      - image_list: lista de rutas relativas ("clase/archivo")
      - label_list: lista de etiquetas (clase)
    """

    df = pd.read_csv(DB_PATH / "db.csv")
    image_list = df["image"].tolist()
    label_list = df["label"].tolist()
    return image_list, label_list


def iter_queries_from_folder(root: Path):

    """
    Genera queries leyendo una estructura tipo root/<clase>/<archivo>

    Devuelve una lista de tuplas:
      (true_label, path_imagen)
    """

    out = []
    for cls in os.listdir(root):
        cls_dir = root / cls
        if cls_dir.is_dir():
            for f in os.listdir(cls_dir):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    out.append((cls, cls_dir / f))
    return out


def iter_queries_from_gallery(image_list, label_list):

    """
    Genera queries a partir de la propia galería (train).
    Se usan todas las imágenes indexadas como queries.
    """

    return [(lab, IMAGES_PATH / rel_path) for rel_path, lab in zip(image_list, label_list)]


def evaluate_split(split_name, queries, label_list, index, extractor_fn, k=5):

    """
    Evalúa un conjunto de queries (train o test) contra el índice FAISS.

    - split_name: "train" o "test"
    - queries: lista de (true_label, ruta)
    - label_list: etiquetas de la galería, para mapear índices->clases
    - index: índice FAISS ya cargado
    - extractor_fn: función que extrae features (1, D)
    - k: top-K a evaluar

    Devuelve un diccionario con prec@k, map@k y número de queries.
    """

    using_gallery_queries = (split_name.lower() == "train")

    prec_sum = 0.0
    map_sum = 0.0
    total = 0

    for true_label, path in queries:
        img = Image.open(path).convert("RGB")   # Cargamos la imagen query

        # Extraemos el vector y lo normalizamos
        q = extractor_fn(img).astype("float32")
        faiss.normalize_L2(q)

        # En TRAIN pedimos k+1 y quitamos el primer resultado (self-match)
        search_k = k + 1 if using_gallery_queries else k
        _, idxs = index.search(q, search_k)

        # Convertimos índices recuperados->etiquetas de clase
        retrieved_labels = [label_list[i] for i in idxs[0]]
        if using_gallery_queries and retrieved_labels:
            retrieved_labels = retrieved_labels[1:]  # quitamos el self-match

        # Acumulamos métricas por query
        prec_sum += precision_at_k(retrieved_labels, true_label, k)
        map_sum += ap_at_k(retrieved_labels, true_label, k)
        total += 1

    # Media sobre todas las queries
    return {
        "prec@k": prec_sum / max(total, 1),
        "map@k": map_sum / max(total, 1),  # mAP@K 
        "n": total,
    }


def main(k=5):

    """
    Entrada:
    - Carga la galería (db.csv)
    - Genera queries para train y test
    - Para cada extractor:
        - carga su índice FAISS
        - evalúa train y test
        - guarda métricas en un DataFrame
    - Imprime tabla final y guarda CSV.
    """

    image_list, label_list = load_gallery()

    train_queries = iter_queries_from_gallery(image_list, label_list)
    test_queries = iter_queries_from_folder(TEST_PATH)

    rows = []
    for name, extractor_fn in EXTRACTORS.items():
        index_path = DB_PATH / f"feat_{name}.index"
        if not index_path.exists():
            print(f"[WARN] Falta índice para {name}: {index_path}")
            continue

        # Cargamos el índice FAISS
        index = faiss.read_index(str(index_path))

        # Evaluamos métricas en train y test
        train_m = evaluate_split("train", train_queries, label_list, index, extractor_fn, k=k)
        test_m  = evaluate_split("test", test_queries, label_list, index, extractor_fn, k=k)

        rows.append({
            "extractor": name,
            f"train_prec@{k}": train_m["prec@k"],
            f"train_mAP@{k}": train_m["map@k"],
            f"test_prec@{k}": test_m["prec@k"],
            f"test_mAP@{k}": test_m["map@k"],
        })

    # Ordenamos por rendimiento en test (mAP@k)
    df = pd.DataFrame(rows).sort_values(by=f"test_mAP@{k}", ascending=False)

    print(f"\n=== TRAIN vs TEST | Precision@{k} & mAP@{k} ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Guardamos un csv
    out_csv = DB_PATH / f"metrics_train_test_prec_map_at{k}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[OK] Guardado en: {out_csv}")


if __name__ == "__main__":
    main(k=5)
