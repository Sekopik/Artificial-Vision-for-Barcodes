import os
import time
import threading
import queue
import shutil
import cv2
import logging
import numpy as np  # <--- ESTO FALTABA
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from config import cfg
from image_processor import ImagePipeline
from database_handler import DatabaseHandler, StatusEnum

# Configuración de Logging
logging.basicConfig(
    filename=cfg.get('PATHS', 'log_file'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class FileProcessor(threading.Thread):
    def __init__(self, job_queue, pipeline, db):
        super().__init__()
        self.job_queue = job_queue
        self.pipeline = pipeline
        self.db = db
        self.daemon = True # Se cierra cuando termina el main

    def safe_read_image(self, path, retries=3, delay=0.5):
        """
        Intenta leer la imagen manejando bloqueos del SO (PermissionError).
        """
        for i in range(retries):
            try:
                if not os.path.exists(path):
                    return None
                    
                # Usamos cv2.imdecode para mejor soporte de caracteres no-ascii en rutas
                with open(path, "rb") as stream:
                    bytes_data = bytearray(stream.read())
                    numpy_array = np.asarray(bytes_data, dtype=np.uint8)
                    img = cv2.imdecode(numpy_array, cv2.IMREAD_UNCHANGED)
                
                if img is None:
                    raise ValueError("Imagen corrupta o formato no soportado")
                return img
            except (PermissionError, OSError):
                time.sleep(delay)
            except Exception as e:
                print(f"[ERROR LECTURA] {path}: {e}")
                break
        return None

    def run(self):
        while True:
            path = self.job_queue.get()
            if path is None: 
                self.job_queue.task_done()
                break

            filename = os.path.basename(path)
            start_time = time.time()
            
            try:
                # 1. Lectura Segura
                image = self.safe_read_image(path)
                
                if image is None:
                    # Si falla la lectura, movemos y registramos error
                    self.move_file(path, cfg.get('PATHS', 'error_folder'))
                    self.log_result(filename, "N/A", 0, StatusEnum.ERROR, 0)
                    # IMPORTANTE: No llamamos a task_done() aquí, se hace en el finally
                    continue 

                # 2. Detección ROI (YOLO)
                roi_image = self.pipeline.detect_roi(image)

                # 3. Preprocesamiento
                processed_img = self.pipeline.preprocess_image(roi_image)
                
                # Guardar debug
                debug_path = os.path.join(cfg.get('PATHS', 'debug_folder'), f"proc_{filename}")
                # Asegurar que la carpeta debug existe (por si acaso)
                os.makedirs(cfg.get('PATHS', 'debug_folder'), exist_ok=True)
                cv2.imwrite(debug_path, processed_img)

                # 4. Extracción OCR
                text, confidence = self.pipeline.extract_text(processed_img)
                
                end_time = time.time()
                duration = end_time - start_time

                # 5. Decisión y Base de Datos
                status = StatusEnum.SUCCESS if text else StatusEnum.FAIL_OCR
                
                self.log_result(filename, text, confidence, status, duration)
                self.db.save_event(filename, text, confidence, status, duration)

                # 6. Mover archivo
                dest_folder = cfg.get('PATHS', 'processed_folder') if text else cfg.get('PATHS', 'error_folder')
                self.move_file(path, dest_folder)

            except Exception as e:
                logging.error(f"Excepción crítica procesando {filename}: {e}")
                print(f"[CRITICAL ERROR] {e}")
                try:
                    self.move_file(path, cfg.get('PATHS', 'error_folder'))
                except:
                    pass
            finally:
                # Marcamos la tarea como completada SIEMPRE, pase lo que pase
                self.job_queue.task_done()

    def move_file(self, src, folder):
        try:
            if not os.path.exists(src):
                return
            
            os.makedirs(folder, exist_ok=True)
            dest = os.path.join(folder, os.path.basename(src))
            
            # Manejar si ya existe
            if os.path.exists(dest):
                base, ext = os.path.splitext(dest)
                dest = f"{base}_{int(time.time())}{ext}"
            shutil.move(src, dest)
        except Exception as e:
            logging.error(f"Error moviendo archivo {src}: {e}")

    def log_result(self, filename, text, conf, status, duration):
        engine = cfg.get('MODULES', 'ocr_engine')
        print(f"[INFO] Archivo: {filename} | Estado: {status.value} | Codigo: {text} | Conf: {conf:.2f} | Engine: {engine} | T: {duration:.2f}s")
        logging.info(f"File: {filename}, Code: {text}, Status: {status.value}")

class NewImageHandler(FileSystemEventHandler):
    def __init__(self, queue):
        self.queue = queue

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
            # Pequeña espera inicial
            time.sleep(0.5) 
            self.queue.put(event.src_path)

def main():
    print("=== SISTEMA DE RECUPERACIÓN DE LECTURAS FALLIDAS ===")
    print(f"Iniciando monitoreo en: {os.path.abspath(cfg.get('PATHS', 'input_folder'))}")
    
    # Inicializar componentes
    try:
        db = DatabaseHandler()
        pipeline = ImagePipeline()
    except Exception as e:
        print(f"[CRITICAL] Error inicializando componentes: {e}")
        return

    # Cola de trabajo
    job_queue = queue.Queue()

    # Iniciar Worker
    worker = FileProcessor(job_queue, pipeline, db)
    worker.start()

    # Iniciar Watchdog
    observer = Observer()
    event_handler = NewImageHandler(job_queue)
    
    input_folder = cfg.get('PATHS', 'input_folder')
    if not os.path.exists(input_folder):
        os.makedirs(input_folder)

    observer.schedule(event_handler, input_folder, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n[SHUTDOWN] Deteniendo sistema...")
    
    observer.join()

if __name__ == "__main__":
    main()