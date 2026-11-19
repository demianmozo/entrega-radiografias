# Clasificación de radiografías de tórax

Primera entrega del trabajo final de Inteligencia Artificial. Se clasifican imágenes de tres categorías: Normal, COVID-19 y Viral Pneumonia.

Abrí [clasificacion_radiografias.ipynb](clasificacion_radiografias.ipynb) para ver la explicación, el código y los resultados guardados. La corrida original entrenó una ResNet-18 con pesos iniciales aleatorios. En 2.274 imágenes de test obtuvo **97,01% de accuracy** y **0,9627 de macro-F1**, frente a **67,24%** y **0,2680** al predecir siempre Normal.

## Ejecutar el cuaderno

Con Python 3.13, creá un entorno desde la carpeta del repositorio:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch==2.11.0+cpu torchvision==0.26.0+cpu --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
```

Abrí esa carpeta en VS Code y elegí `.venv/bin/python` como kernel del cuaderno. Las tablas de auditoría, las curvas y las métricas se recalculan con los CSV incluidos. Las celdas de imágenes y de la prueba de entrenamiento requieren el dataset.

## Descargar las imágenes

Descargá [COVID-19 Radiography Database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database/data) y descomprimí la carpeta `COVID-19_Radiography_Dataset` dentro de `data/raw/`. Por ejemplo, `data/raw/COVID-19_Radiography_Dataset/Normal/images/` debe contener los PNG de la clase Normal.

Con los datos disponibles se pueden ejecutar todas las celdas. Para regenerar la auditoría y la partición se puede correr `.venv/bin/python src/audit.py`. El reentrenamiento completo se activa desde el cuaderno y llevó unos 30 minutos efectivos en la CPU del servidor.

Las métricas describen esta colección. Se separaron grupos de duplicados detectados y los datos carecen de IDs de paciente verificables.
