# Artificial Vision for Barcodes

Sistema industrial modular escrito en **Python 3.11** para la recuperación de lecturas fallidas en líneas de paquetería. 

Este software monitorea una carpeta en tiempo real, procesa imágenes con defectos (brillos, baja resolución, etiquetas matriciales) utilizando técnicas avanzadas de preprocesamiento (CLAHE, Dilatación) y extrae códigos numéricos mediante **RapidOCR (ONNX)** o **DocTR**.

## Características

- **Arquitectura Asíncrona:** Implementación Producer-Consumer con `Watchdog` y `Queue` para no bloquear el I/O.
- **Preprocesamiento "Quirúrgico":** 
    - Eliminación de brillos en plásticos (CLAHE).
    - Reconstrucción de fuentes de puntos (Dilatación morfológica).
    - Upscaling inteligente.
- **Motores OCR Soportados:**
    - `RapidOCR` (Por defecto, ligero y rápido vía ONNX).
    - `DocTR` (Para casos complejos con Deep Learning).
- **Persistencia:** Registro automático de eventos en SQLite mediante SQLAlchemy.
- **Tolerancia a Fallos:** Sistema de reintentos para lectura de archivos bloqueados por el SO.

## Requisitos

- Python 3.11
- Entorno Windows/Linux

## Instalación

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/TU_USUARIO/Artificial-Vision-for-Barcodes.git
   cd Artificial-Vision-for-Barcodes

2. **Crear entorno virtual e instalar dependencias**

    # Windows
    py -3.11 -m venv venv
    .\venv\Scripts\activate

    # Instalar librerías
    pip install -r requirements.txt

3. **Configuracion**

    [MODULES]
    ocr_engine = RAPIDOCR      ; Opciones: RAPIDOCR, DOCTR
    enable_yolo_detection = false ; Activar si se dispone de modelo .pt
    enable_preprocessing = true   ; Recomendado para etiquetas reales

4. **USO**

    Ejecutar = python main.py

    El sistema creará automáticamente las carpetas:
    input_images/: Arrastra aquí tus imágenes para procesar.
    processed_images/: Imágenes leídas con éxito.
    error_images/: Imágenes fallidas.
    debug_output/: Visualización del preprocesamiento (útil para debug).


**ESTRUCTURA DEL PROYECTO**

    - Estructura del Proyecto
    - main.py: Orquestador y manejo de hilos (Watchdog + Workers).
    - image_processor.py: Núcleo de Visión Artificial (OpenCV + OCR).
    - database_handler.py: ORM para SQLite.
    - config.py: Singleton para gestión de configuración.
