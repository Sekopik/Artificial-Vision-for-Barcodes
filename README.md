# Artificial Vision for Barcodes

Sistema industrial modular escrito en **Python 3.11** para la recuperación de lecturas fallidas en líneas de paquetería y logística.

Este software monitorea una carpeta en tiempo real, procesa imágenes con defectos (brillos, baja resolución, etiquetas matriciales) utilizando técnicas avanzadas de visión artificial y extrae información mediante modelos de **Deep Learning (YOLO)** y motores **OCR**.

## 🧠 Características

- **Arquitectura Asíncrona:** Implementación Producer-Consumer con `Watchdog` y `Queue` para maximizar el rendimiento I/O.
- **Detección Inteligente (YOLOv8):** Localización precisa de etiquetas mediante modelo orientado (OBB) en `best.pt`.
- **Preprocesamiento \"Quirúrgico\":**
    - Eliminación de brillos en plásticos (CLAHE).
    - Reconstrucción de fuentes de puntos (Dilatación morfológica).
    - Upscaling inteligente para códigos pequeños.
- **Motores OCR Soportados:**
    - `RapidOCR` (Optimizado para velocidad vía ONNX).
    - `DocTR` (Para documentos complejos).
- **Persistencia:** Registro automático de eventos en SQLite mediante SQLAlchemy.
- **Tolerancia a Fallos:** Sistema de reintentos automático para gestión de archivos bloqueados por el SO.

## 📋 Requisitos del Sistema

El proyecto requiere estrictamente **Python 3.11** por compatibilidad de librerías de tensores y visión artificial.

### Configuración en Ubuntu 24.04 LTS (o superior)
Dado que Ubuntu 24.04 trae versiones más nuevas de Python, es necesario instalar la versión 3.11 manualmente:

```bash
sudo apt update
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.11 python3.11-venv python3.11-dev
```

## ⚙️ Instalación

Sigue estos pasos para desplegar el entorno de desarrollo en Linux/Ubuntu:

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/Sekopik/Artificial-Vision-for-Barcodes.git
   cd Artificial-Vision-for-Barcodes
   ```

2. **Crear el entorno virtual (VENV):**
   Es fundamental usar el binario de Python 3.11 explícitamente:
   ```bash
   python3.11 -m venv venv
   ```

3. **Activar el entorno:**
   ```bash
   source venv/bin/activate
   ```
   *(Deberías ver `(venv)` al inicio de tu línea de comandos).*

4. **Instalar dependencias:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## 🔧 Configuración (`config.ini`)

El sistema se controla mediante el archivo `config.ini`. Asegúrate de que los parámetros coincidan con tu entorno actual:

```ini
[PATHS]
input_folder = ./input_images
processed_folder = ./processed_images
error_folder = ./error_images
debug_folder = ./debug_output
log_file = ./system.log

[DATABASE]
db_file = sqlite:///shipping_data.db

[MODULES]
ocr_engine = RAPIDOCR
; Activa la detección previa con YOLO (requiere best.pt)
enable_yolo_detection = true
yolo_model_path = best.pt

[OCR_PARAMS]
; Confianza de detección de etiqueta (YOLO)
yolo_conf = 0.35
yolo_imgsz = 1024

; Confianza mínima para aceptar un caracter OCR
ocr_min_confidence = 0.5
```

## 🚀 Uso

1. **Ejecutar el programa principal:**
   ```bash
   python main.py
   ```
   *Al iniciar, el sistema creará automáticamente las carpetas de trabajo (`input_images`, `processed_images`, etc.) si no existen.*

2. **Procesar imágenes:**
   Arrastra o copia tus imágenes en la carpeta `input_images/`. El sistema las detectará automáticamente.

3. **Detener el sistema:**
   *   **Opción A (Visual):** Si se abre una ventana de visualización, pulsa la tecla **`q`** o **`Esc`**.
   *   **Opción B (Terminal):** Pulsa **`Ctrl + C`** en la terminal. Si el proceso está muy ocupado, usa `Ctrl + \\`.

## 📂 Estructura del Proyecto

- `main.py`: Orquestador principal y gestión de hilos (Watchdog + Workers).
- `image_processor.py`: Núcleo de Visión Artificial (OpenCV + YOLO + OCR).
- `database_handler.py`: ORM para gestión de base de datos SQLite.
- `config.py`: Singleton para la gestión centralizada de la configuración.
- `best.pt`: Pesos del modelo YOLO entrenado para detección de etiquetas.