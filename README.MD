# CBIR 

INTEGRANTES DEL GRUPO:
- Natalia Figal Izquierdo
- Gloria Domingo de Loresecha

Este repositorio ya incluye la base de datos (database/db.csv) y los índices FAISS (`database/feat_*.index), así que para probar el sistema no hace falta reconstruir nada, basta con ejecutar la app.

Nota: durante el desarrollo se intentó trabajar con Python 3.12, pero se produjeron problemas de compatibilidad (errores de DLL) con algunas librerías de visión y deep learning en Windows.  
Para poder continuar el proyecto y asegurar su correcto funcionamiento, decidimos utilizar Miniconda con Python 3.10, donde el sistema funciona de forma estable.

1) Instalar dependencias: pip install -r requirements.txt

2) Ejecutar la aplicación: streamlit run app.py


Estructura básica:
- images/: galería TRAIN organizada por clases
- images_test/: conjunto de TEST organizado por clases
- database/: db.csv e índices FAISS 



